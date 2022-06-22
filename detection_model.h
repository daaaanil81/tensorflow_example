#ifndef __DETECTION_MODEL_H__
#define __DETECTION_MODEL_H__

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using tensorflow::Flag;
using tensorflow::Scope;
using tensorflow::Tensor;
using tensorflow::GraphDef;
using tensorflow::SavedModelBundle;
using tensorflow::SessionOptions;
using tensorflow::ClientSession;
using tensorflow::RunOptions;
using tensorflow::Output;
using tensorflow::Status;
using tensorflow::DT_FLOAT;
using tensorflow::DT_UINT8;
using tensorflow::DT_STRING;
using tensorflow::TensorShape;

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
    Status ImageToTensor(Tensor& imageTensor);
    void Predict(const Tensor& imageTensor, std::vector<Tensor>& predictions);
public:
    DetectionModel(const std::string& path_to_model);

    void Testing(const std::string& path_to_image);
    void Testing(cv::Mat& image);
};

#endif /* __DETECTION_MODEL_H__ */
