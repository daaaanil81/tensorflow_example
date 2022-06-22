#include "detection_model.h"

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
    /* auto cast_image = Cast(_root.WithOpName("cast"), image_reader, DT_FLOAT); */
    auto cast_image = Cast(_root.WithOpName("cast"), image_reader, DT_UINT8);
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

Status DetectionModel::ImageToTensor(Tensor& imageTensor) {
    using namespace tensorflow::ops;

    GraphDef graph;
    std::vector<Tensor> vecTensors;

    auto output_of_graph = ExpandDims(_root.WithOpName("dim"), imageTensor,
            _expand_dims_axis);

	TF_RETURN_IF_ERROR(_root.ToGraphDef(&graph));
	ClientSession session(_root);

	auto run_status = session.Run({output_of_graph}, &vecTensors);
	if (!run_status.ok()){
		printf("Error in running session \n");
	}

    LOG(INFO) << imageTensor.DebugString();
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

    Predict(imageTensor, predictions);
}

void DetectionModel::Testing(cv::Mat& image) {
    const clock_t begin_time = clock();

    Tensor imageTensor(DT_UINT8, TensorShape({1, image.rows, image.cols, 3});
    std::cout << "imageTensor is empty: " << std::endl << imageTensor.DebugString() << std::endl;
    uint8_t *p = inputImg.flat<tensorflow:uint8_t>().data();
    cv::Mat tensorMatImage(image.rows, image.cols, CV_8UC3, p);

    image.convertTo(tensorMatImage, CV_8UC3);

    auto status = ImageToTensor(imageTensor);
    if (!status.ok()) {
		LOG(ERROR) << status.ToString();
        throw std::runtime_error(status.ToString());
    }

    LOG(INFO) << "Time of Convert Mat to tensor: "
              << float(clock() - begin_time) / CLOCKS_PER_SEC;

    Predict(imageTensor, predictions);
}

void DetectionModel::Predict(const Tensor& imageTensor,
    std::vector<Tensor>& predictions) {

    const clock_t begin_time = clock();
    /* status = _model.GetSession()->Run({{_input_layer, imageTensor}}, */
    /*         _output_layers, {}, &predictions); */

    auto status = _model.GetSession()->Run({{input_nodes, imageTensor}},
            output_nodes, {}, &predictions);
    if (!status.ok()) {
		LOG(ERROR) << status.ToString();
        throw std::runtime_error(status.ToString());
    }

    LOG(INFO) << "Time of Predict: "
              << float(clock() - begin_time) / CLOCKS_PER_SEC;

    LOG(INFO) << "Run is successfully. Predictions: " << predictions.size();

    for (auto tensor : predictions) {
        std::cout << tensor.DebugString() << std::endl;
    }
}
