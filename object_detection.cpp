/*
 * This program is used for object detecting by using model from link:
 * https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
 *
 * Program has class with hard code names of layers inside class.
 * So, this class can work only with that model.
 * */

#include <iostream>
#include <string>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using tensorflow::Flag;
using tensorflow::Scope;
using tensorflow::Tensor;
using tensorflow::SavedModelBundle;
using tensorflow::SessionOptions;
using tensorflow::ClientSession;
using tensorflow::RunOptions;
using tensorflow::Output;
using tensorflow::Status;
using tensorflow::DT_FLOAT;
using tensorflow::DT_UINT8;
using tensorflow::DT_STRING;

class DetectionModel {
private:
    Scope _root;
    std::string _path_to_model;

    SavedModelBundle _model;
    SessionOptions session_options;
	RunOptions run_options;

    int _image_channels = 3; // RGB - 3, Gray - 2
    int _expand_dims_axis = 0; // Index for inserting

    Output _input_of_graph;
    Output _output_of_graph;

    std::string _input_layer = "hub_input/image_tensor:0";
    std::vector<std::string> _output_layers = {{
        "hub_input/strided_slice:0",
        "hub_input/index_to_string_Lookup:0",
        "hub_input/strided_slice_2:0",
        "hub_input/index_to_string_1_Lookup:0",
        "hub_input/strided_slice_1:0"
    }};

    std::string input_nodes = "serving_default_input_tensor:0";
    std::vector<std::string> output_nodes = {{
        "StatefulPartitionedCall:0", //detection_anchor_indices
		"StatefulPartitionedCall:1", //detection_boxes
		"StatefulPartitionedCall:2", //detection_classes
		"StatefulPartitionedCall:3", //detection_multiclass_scores
		"StatefulPartitionedCall:4", //detection_scores
		"StatefulPartitionedCall:5"  //num_detections
    }};

    Status CreateGraphForImage();
    Status ImageToTensor(const std::string& path_to_image, Tensor& imageTensor);
public:
    DetectionModel(const std::string& path_to_model);

    void Testing(const std::string& path_to_image);
};

DetectionModel::DetectionModel(const std::string& path_to_model) :
    _root(Scope::NewRootScope()), _path_to_model(path_to_model) {
	auto status = tensorflow::LoadSavedModel(session_options, run_options,
        path_to_model, {}, &_model);
        /* path_to_model, {"serve"}, &_model); */
	if (status.ok()) {
		LOG(INFO) << "Model was loaded successfully...";
	} else {
		LOG(ERROR) << "Error in loading model";
        throw std::runtime_error(status.ToString());
	}

    status = CreateGraphForImage();
    if (!status.ok()) {
		LOG(ERROR) << "Failed to create graph";
        throw std::runtime_error(status.ToString());
    }
}

Status DetectionModel::CreateGraphForImage() {
    using namespace tensorflow::ops;

    _input_of_graph = Placeholder(_root.WithOpName("input"), DT_STRING);
    auto file_reader = ReadFile(_root.WithOpName("file_reader"), _input_of_graph);
    auto image_reader = DecodeJpeg(_root.WithOpName("image_decoder"),
            file_reader, DecodeJpeg::Channels(_image_channels));
    auto cast_image = Cast(_root.WithOpName("cast"), image_reader, DT_FLOAT);
    /* auto cast_image = Cast(_root.WithOpName("cast"), image_reader, DT_UINT8); */
    _output_of_graph = ExpandDims(_root.WithOpName("dims"), cast_image,
            _expand_dims_axis);

    return _root.status();
}

Status DetectionModel::ImageToTensor(const std::string& path_to_image,
    Tensor& imageTensor) {

    using namespace tensorflow;

    std::vector<Tensor> vecTensors;

    if (!str_util::EndsWith(path_to_image, ".jpg") &&
        !str_util::EndsWith(path_to_image, ".jpeg")) {
		LOG(ERROR) << "Image must be jpeg/jpg encoded";
        return errors::InvalidArgument("Image must be jpeg/jpg encoded");
    }

    ClientSession session(_root);

    TF_RETURN_IF_ERROR(session.Run({{_input_of_graph, path_to_image}},
            {_output_of_graph}, &vecTensors));

    imageTensor = std::move(vecTensors[0]);

    LOG(INFO) << imageTensor.DebugString();

    return Status::OK();
}

void DetectionModel::Testing(const std::string& path_to_image) {
    std::vector<Tensor> predictions;
    Tensor imageTensor;

    auto status = ImageToTensor(path_to_image, imageTensor);
    if (!status.ok()) {
		LOG(ERROR) << status.ToString();
        throw std::runtime_error(status.ToString());
    }

    LOG(INFO) << "Image was loaded";

    status = _model.GetSession()->Run({{_input_layer, imageTensor}},
            _output_layers, {}, &predictions);

    /* status = _model.GetSession()->Run({{input_nodes, imageTensor}}, */
    /*         output_nodes, {}, &predictions); */
    if (!status.ok()) {
		LOG(ERROR) << status.ToString();
        throw std::runtime_error(status.ToString());
    }

    LOG(INFO) << "Run is successfully. Predictions: " << predictions.size();

    for (auto tensor : predictions) {
        std::cout << tensor.DebugString() << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string path_to_image;
    std::string path_to_model;

    std::vector<Flag> flag_list = {
        Flag("image", &path_to_image, "path of image to be processed"),
        Flag("model", &path_to_model, "path of model to be processed"),
    };

    std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    if (!parse_result) {
        LOG(ERROR) << usage;
        return -1;
    }

    if (path_to_image.empty()) {
        LOG(ERROR) << "Path of image is empty!!!";
        return -1;
    }
    LOG(INFO) << "Path of image: " << path_to_image;

    if (path_to_model.empty()) {
        LOG(ERROR) << "Path of model is empty!!!";
        return -1;
    }
    LOG(INFO) << "Path of model: " << path_to_model;

    try {
        DetectionModel model(path_to_model);
        model.Testing(path_to_image);
    } catch (const std::exception& e) {
        LOG(ERROR) << e.what();
        return -1;
    }

    return 0;
}
