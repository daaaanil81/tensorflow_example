#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <fstream>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::Scope;
using tensorflow::GraphDef;
using tensorflow::Output;
using tensorflow::SessionOptions;
using tensorflow::ClientSession;
using tensorflow::Session;
using tensorflow::RunOptions;
using tensorflow::SavedModelBundle;

class Model {
private: /* Variables */
    /* scope for graph */
    Scope _root;

    /* Layers in model */
    std::string _input_layer;
    std::string _output_layer;

    /* Labels */
    std::vector<std::string> _labels;

    /* Read SavedModel type */
    SessionOptions _session_options;
    RunOptions _run_options;
    SavedModelBundle _bundle;

    /* Variables for graph */
    Output _input_of_graph;
    Output _output_of_graph;

    int _image_channels = 3; // RGB - 3, Gray - 2
    int _expand_dims_axis = 0; // Index for inserting
    int _input_height = 96;
    int _input_width = 96;
    float _convert_data_value = 255.f;

private: /* Functions */
    Status ReadLabelsFile(const std::string& file_name);
    Status CreateGraphForImage();
    Status ReadImageToTensor(const std::string& file_name, Tensor& out_tensors);
    Status GetTopLabels(const std::vector<Tensor>& outputs, Tensor* indices,
            Tensor* scores);
    void PrintTopLabels(Tensor& indexes, Tensor& scores);

public:
    /* Read only SavedModel type.
     * Have 2 folders, "assets" and "variables", and one file "save_model.pb".
     * */
    Model(const std::string& model_path, const std::string& file_labels,
            const std::string& input_layer, const std::string& output_layer);

    std::tuple<int32_t, float> Testing(const std::string& file_path);
};

Model::Model(const std::string& model_path, const std::string& file_labels,
        const std::string& input_layer, const std::string& output_layer)
        : _root(Scope::NewRootScope()),  _input_layer(input_layer),
        _output_layer(output_layer) {
    /* SessionOption - configuration information for a Session.
     * export_dir - the path of directory.
     * tags("serve") - used at SavedModel build time.
     * Bundle - model bundle.
     * */
    auto status = tensorflow::LoadSavedModel(_session_options, _run_options,
            model_path, {"serve"}, &_bundle);
    if (!status.ok()) {
        throw std::runtime_error(status.ToString());
    }

    status = ReadLabelsFile(file_labels);
    if (!status.ok()) {
        throw std::runtime_error(status.ToString());
    }

    status = CreateGraphForImage();
    if (!status.ok()) {
        throw std::runtime_error(status.ToString());
    }
}

Status Model::ReadLabelsFile(const std::string& file_name) {
    using namespace tensorflow;

    std::string line;

    std::ifstream file(file_name);
    if (!file) {
        return errors::NotFound("Labels file ", file_name, " not found.");
    }

    while (std::getline(file, line)) {
        _labels.push_back(line);
    }

    return Status::OK();
}

Status Model::CreateGraphForImage() {
    using namespace tensorflow;

    /* Add receiving directory
     * Return tensorflow::Output
     * */
    _input_of_graph = ops::Placeholder(_root.WithOpName("input"), DT_STRING);

    /* Read file
     * Return tensorflow::Output
     * */
    auto file_reader = ops::ReadFile(_root.WithOpName("file_reader"),
            _input_of_graph);

    /* Receive Jpeg file
     * Return tensorflow::Output
     * */
    auto image_reader = ops::DecodeJpeg(_root.WithOpName("image_decoder"),
            file_reader, ops::DecodeJpeg::Channels(_image_channels));

    /* Casting data to float type
     * Return tensorflow::Output
     * */
    auto cast_image = ops::Cast(_root.WithOpName("cast"), image_reader,
            DT_FLOAT);

    /* Add number of banch
     * (height, width, channels) -> (banch, height, width, channels)
     * Return tensorflow::Output
     * */
    auto dims = ops::ExpandDims(_root.WithOpName("dims"),
            cast_image, _expand_dims_axis);

    /* Resize images
     * Return tensorflow::Output
     * */
    _output_of_graph = ops::ResizeBilinear(_root.WithOpName("resize"), dims,
            ops::Const(_root.WithOpName("size"), {_input_height, _input_width}));

    /* Delete datas on convert_data_value
     * Return tensorflow::Output
     * Normalized isn't used because rescaling is used in graph of model.
     * */
    /* _output_of_graph = Div(_root.WithOpName("div"), resize, */
    /*         {_convert_data_value}); */

    return _root.status();
}

Status Model::ReadImageToTensor(const std::string& file_name,
        Tensor& out_tensor) {
    using namespace tensorflow;

    std::vector<Tensor> out_tensors;

    if (!str_util::EndsWith(file_name, ".jpg") &&
        !str_util::EndsWith(file_name, ".jpeg")) {
        return errors::InvalidArgument("Image must be jpeg/jpg encoded");
    }

    ClientSession session(_root);

    TF_RETURN_IF_ERROR(session.Run({{_input_of_graph, file_name}},
            {_output_of_graph}, &out_tensors));

    out_tensor = out_tensors[0]; // shallow copy

    return Status::OK();
}


std::tuple<int32_t, float> Model::Testing(const std::string& file_path) {
        Tensor out_tensor;
    std::vector<Tensor> outputs;
    Tensor indexes, scores;

    auto status = ReadImageToTensor(file_path, out_tensor);
    if (!status.ok()) {
        throw std::runtime_error(status.ToString());
    }

    status = _bundle.GetSession()->Run({{_input_layer, out_tensor}},
            {_output_layer}, {}, &outputs);
    if (!status.ok()) {
        throw std::runtime_error(status.ToString());
    }

    status = GetTopLabels(outputs, &indexes, &scores);
    if (!status.ok()) {
        throw std::runtime_error(status.ToString());
    }

    tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
    tensorflow::TTypes<int32_t>::Flat indexes_flat = indexes.flat<int32_t>();

    PrintTopLabels(indexes, scores);

    return std::make_tuple(indexes_flat(0), scores_flat(0));
}

Status Model::GetTopLabels(const std::vector<Tensor>& inputs,
        Tensor* indexes, Tensor* scores) {
    using namespace tensorflow;

    GraphDef graph;
    std::vector<Tensor> outputs;
    std::unique_ptr<Session> session(NewSession(SessionOptions()));
    auto tmp_root = Scope::NewRootScope();

    if (inputs.size() == 0) {
        return errors::NotFound("No found output from model");
    }

    ops::TopK(tmp_root.WithOpName("top_k"), inputs[0], (int32_t)_labels.size());

    TF_RETURN_IF_ERROR(tmp_root.ToGraphDef(&graph));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({}, {"top_k:0", "top_k:1"}, {}, &outputs));

    if (outputs.size() == 0) {
        return errors::NotFound("No found result from top_k");
    }

    *scores = outputs[0];
    *indexes = outputs[1];

    return Status::OK();
}

void Model::PrintTopLabels(Tensor& indexes, Tensor& scores) {
    tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
    tensorflow::TTypes<int32_t>::Flat indexes_flat = indexes.flat<int32_t>();

    for (size_t pos = 0; pos < _labels.size(); ++pos) {
        const int label_index = indexes_flat(pos);
        const float score = scores_flat(pos);
        LOG(INFO) << _labels[label_index] << " (" << label_index << "): " << score;
    }
}

int main() {
    int32_t index;
    float score;
    std::string path_to_model = "<path_to_model>";
    std::string testing_file = "<path_to_image>";
    std::string file_labels = "<path_to_file_of_labels";
    std::string input_model_layer = "serving_default_rescaling_input:0";
    std::string output_model_layer = "StatefulPartitionedCall:0";

    try {
        Model model(path_to_model, file_labels, input_model_layer,
                output_model_layer);

        std::tie(index, score) = model.Testing(testing_file);

        std::cout << "Result: index: " << index
            << " score: " << score << std::endl;
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
