


#include <opencv2/opencv.hpp>  
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/lrn_layer.hpp>
#include<caffe/layers/concat_layer.hpp>
#include<caffe/layers/relu_layer.hpp>
#include<caffe/layers/pooling_layer.hpp>
#include<caffe/layers/inner_product_layer.hpp>
#include<caffe/layers/softmax_layer.hpp>//要添加包含各个层头文件
#include<caffe/layers/concat_layer.hpp>
#include<caffe/layers/eltwise_layer.hpp>
#include<caffe/layers/silence_layer.hpp>
#include<caffe/layers/exp_layer.hpp>
#include<caffe/layers/power_layer.hpp>
#include<caffe/layers/batch_norm_layer.hpp>
#include<caffe/layers/scale_layer.hpp>




#include <caffe/blob.hpp>
#include <caffe/solver.hpp>
//#include "HolidayCNN_proto.pb.h"
// these need to be included after boost on OS X  
#include <string>  // NOLINT(build/include_order)  
#include <vector>  // NOLINT(build/include_order)  
#include <fstream>  // NOLINT  

//#include "ParseToHolidayLayerDetail.h"

//#include "HolidayModelStorage.h"
//#include "ReadFromHolidayLayer.h"

#ifndef CPU_ONLY

#pragma comment("cuda.lib");
#pragma comment("cublas.lib");
#pragma comment("cublas_device.lib");
#pragma comment("cudart.lib");
#pragma comment("curand.lib");
#endif


//#include "ConvertTools.h"


#define NetF float


static void CheckFile(const std::string& filename) {
	std::ifstream f(filename.c_str());
	if (!f.good()) {
		f.close();
		throw std::runtime_error("Could not open file " + filename);
	}
	f.close();
}

template <typename Dtype>
caffe::Net<Dtype>* Net_Init_Load(
	std::string param_file, std::string pretrained_param_file, caffe::Phase phase)
{
	caffe::Caffe::Get().set_mode(caffe::Caffe::Get().CPU);
	CheckFile(param_file);
	CheckFile(pretrained_param_file);

	//param_file--proto文件名 pretrained_param_file---model file name
	caffe::Net<Dtype>* net(new caffe::Net<Dtype>(param_file, phase));

	net->CopyTrainedLayersFrom(pretrained_param_file);



	return net;
}

bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

int PredictSingleFile(caffe::Net<NetF>* _net, std::string image_filename, cv::Size image_resize_size, std::string output_feature_name, std::string feature_file_name)
{
	
	do
	{
		if (!_net)
		{
			break;
		}
		caffe::MemoryDataLayer<NetF> *m_layer_ = (caffe::MemoryDataLayer<NetF> *)_net->layers()[0].get();
		m_layer_->set_batch_size(1);
		cv::Mat src1;
		src1 = cv::imread(image_filename);

		if (src1.empty())
		{
			break;
		}
		cv::Mat rszimage;
		////  need to resize the input image to your size  
		cv::resize(src1, rszimage, image_resize_size);

		/*for (int row = 0; row < rszimage.rows; row++)
		{
			for (int col = 0; col < rszimage.cols; col++)
			{
				rszimage.at<cv::Vec3b>(row, col) = cv::Vec3b(128, 128, 128);
			}
		}*/

		std::vector<cv::Mat> dv = { rszimage }; // image is a cv::Mat, as I'm using #1416  
		std::vector<int> dvl = { 0 };
		
		//m_layer_->AddMatVector(dv, dvl);
		//float loss = 0.0;
		//_net->Forward(&loss);
		
		m_layer_->AddMatVector(dv, dvl);
	
		int64_t start = cv::getTickCount();
		float loss = 0.0;
		int run_length = 10;
		for (int i = 0; i < run_length; i++)
		{
			_net->Forward(&loss);
		}
		
		int64_t end = cv::getTickCount();
		double time1 = (end - start) / cv::getTickFrequency() / 10 * 1000;

		std::cout << "all" << ":" << time1 << "ms\n";
		//caffe::ConvolutionParameter con1 = _net->layers()[0];

		caffe::Blob<NetF>* prob1 = _net->output_blobs()[0];
		
		boost::shared_ptr<caffe::Blob<NetF>> data = _net->blob_by_name(output_feature_name);
		boost::shared_ptr<caffe::Blob<NetF>> conv1 = _net->blob_by_name("conv1");
		boost::shared_ptr<caffe::Blob<NetF>> pool5 = _net->blob_by_name("pool5");
		boost::shared_ptr<caffe::Blob<NetF>> prob = _net->blob_by_name("prob");
		boost::shared_ptr<caffe::Blob<NetF>> fc8 = _net->blob_by_name("fc8");
		boost::shared_ptr<caffe::Blob<NetF>> norm1 = _net->blob_by_name("InnerProduct1");


		boost::shared_ptr<caffe::Blob<NetF>> out_blob = data;

		std::fstream fs2(feature_file_name, std::ios::out);
		//int count = player2->blobs()[0]->num();
		for (int n = 0; n < out_blob->num(); n++)
		{
			for (int c = 0; c <out_blob->channels(); c++)
			{
				for (int i = 0; i < out_blob->height(); i++)
				{
					for (int j = 0; j < out_blob->width(); j++)
					{
						NetF value1 = out_blob->data_at(n, c, i, j);
						fs2 << value1 << "\t";
					}
					fs2 << "\n";
				}

			}

		}
		fs2.close();

	} while (0);

	return 0;
}

int readLabel(std::string label_file, std::vector<std::string>& labels_)
{
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	std::string line;
	while (std::getline(labels, line))
		labels_.push_back(std::string(line));
	labels.close();
	return 0;
}

int PredictSingleFile(caffe::Net<NetF>* _net, std::string image_filename, cv::Size image_resize_size,const std::vector<std::string>& label_vector, std::string& class_result)
{

	do
	{
		if (!_net)
		{
			break;
		}
		caffe::MemoryDataLayer<NetF> *m_layer_ = (caffe::MemoryDataLayer<NetF> *)_net->layers()[0].get();
		m_layer_->set_batch_size(1);

		cv::Mat src1;
		src1 = cv::imread(image_filename);

		if (src1.empty())
		{
			break;
		}
		cv::Mat rszimage;
		////  need to resize the input image to your size  
		cv::resize(src1, rszimage, image_resize_size);
		std::vector<cv::Mat> dv = { rszimage }; // image is a cv::Mat, as I'm using #1416  
		std::vector<int> dvl = { 0 };
		m_layer_->AddMatVector(dv, dvl);
		float loss = 0.0;
		_net->Forward(&loss);

		m_layer_->AddMatVector(dv, dvl);

		double dft0 = cvGetTickCount();
		loss = 0.0;
		int run_length = 10;
		for (int i = 0; i < run_length; i++)
		{
			_net->Forward(&loss);
		}

		double dft1 = cvGetTickCount();
		double dffreq = cvGetTickFrequency();
		std::cout << "all" << ":" << (dft1 - dft0) / dffreq / 1000 / run_length << "ms\n";

		caffe::Blob<NetF>* prob1 = _net->output_blobs()[0];
		boost::shared_ptr<caffe::Blob<NetF>> conv1 = _net->blob_by_name("conv1");
		boost::shared_ptr<caffe::Blob<NetF>> pool5 = _net->blob_by_name("pool5");
		boost::shared_ptr<caffe::Blob<NetF>> prob = _net->blob_by_name("prob");

		boost::shared_ptr<caffe::Blob<NetF>> out_blob = prob;
		caffe::Blob<NetF>* output_layer = _net->output_blobs()[1];
		
		std::vector<float> result_float;

		std::vector<int> out_position;
		out_position.push_back(0);
		out_position.push_back(0);

		for (int i = 0; i<out_blob->shape()[1]; i++)
		{
			out_position[1] = i;
			NetF value1 = out_blob->data_at(out_position);
			result_float.push_back(value1);
		}
		
		//const float* begin = output_layer->cpu_data();
		//const float* end = begin + output_layer->channels();
		//result_float = std::vector<float>(begin, end);

		std::vector<int> maxN = Argmax(result_float, 5);

		class_result = label_vector[maxN[0]];

	} while (0);

	return 0;
}

template<typename Dtype>
int GetConvolutionLayerParam(caffe::Layer<Dtype>* inputLayer, std::vector<size_t>& param_vector, Dtype*& value, Dtype*& bias_value)
{
	caffe::ConvolutionLayer<Dtype>* pconvolutionlayer = (caffe::ConvolutionLayer<Dtype>*)(inputLayer);

	const caffe::ConvolutionParameter& param = pconvolutionlayer->layer_param().convolution_param();
	size_t kernel_width, kernel_height;
	size_t stride_width, stride_height;
	size_t pad_height, pad_width;
	size_t kernel_number, channel;

	param_vector.resize(11, 0);

	int group_ = param.group();

	param_vector[1] = channel = pconvolutionlayer->blobs()[0]->channels();
	param_vector[0] = kernel_number = param.num_output();
	//p

	if (param.kernel_size_size())
	{
		param_vector[2] = param_vector[3] = kernel_height = kernel_width = param.kernel_size(0);
	}
	if (param.stride_size())
	{
		param_vector[4] = param_vector[5] = stride_height = stride_width = param.stride(0);
	}
	if (param.has_stride_w())
	{
		param_vector[5] = stride_width = param.stride_w();
	}
	if (param.has_stride_h())
	{
		param_vector[4] = stride_height = param.stride_h();
	}
	if (0 == param_vector[4])
	{
		param_vector[4] = 1;
	}
	if (0 == param_vector[5])
	{
		param_vector[5] = 1;
	}
	if (param.pad_size())
	{
		param_vector[6] = param_vector[7] = pad_height = pad_width = param.pad(0);
	}

	if (param.has_kernel_h() || param.has_kernel_w())
	{
		param_vector[2] = kernel_height = param.kernel_h();
		param_vector[3] = kernel_width = param.kernel_w();
	}


	if (param.has_pad_w())
	{
		param_vector[7] = pad_width = param.pad_w();
	}
	if (param.has_pad_h())
	{
		param_vector[6] = pad_height = param.pad_h();
	}

	int dilation_size = param.dilation_size();
	if (1 == dilation_size)
	{
		param_vector[8] = param.dilation(0);
		param_vector[9] = param.dilation(0);
	}
	else if (2 == dilation_size)
	{
		param_vector[8] = param.dilation(0);
		param_vector[9] = param.dilation(1);
	}
	else
	{
		param_vector[8] = 1;
		param_vector[9] = 1;
	}

	if (param.bias_term())
	{
		bias_value = const_cast<Dtype*>(pconvolutionlayer->blobs()[1]->cpu_data());
		param_vector[param_vector.size() - 1] = 1;
	}
	else
	{
		bias_value = nullptr;
		param_vector[param_vector.size() - 1] = 0;
	}
	const Dtype* point_data_start = pconvolutionlayer->blobs()[0]->cpu_data();
	Dtype* p_kernel_value = new Dtype[param_vector[0] * param_vector[1] * param_vector[2] * param_vector[3]];
	Dtype* p_start = p_kernel_value;

	int blob_kenerl_widths = pconvolutionlayer->blobs()[0]->width();
	int blob_kenerl_heights = pconvolutionlayer->blobs()[0]->height();
	int blob_kenerl_channels = pconvolutionlayer->blobs()[0]->channels();
	int blob_kenerl_numbers = pconvolutionlayer->blobs()[0]->num();
	int group_offset = blob_kenerl_widths*blob_kenerl_heights*blob_kenerl_channels*kernel_number / group_;
	for (int n = 0; n < pconvolutionlayer->blobs()[0]->num(); n++)
	{
		for (int c = 0; c < pconvolutionlayer->blobs()[0]->channels(); c++)
		{
			for (int i = 0; i < pconvolutionlayer->blobs()[0]->height(); i++)
			{
				for (int j = 0; j < pconvolutionlayer->blobs()[0]->width(); j++)
				{
					//int offset = ((n * blob_kenerl_channels + c) * blob_kenerl_heights + i) * blob_kenerl_widths + j;
					//*p_start = point_data_start[offset];
					*p_start = pconvolutionlayer->blobs()[0]->data_at(n, c, i, j);
					p_start++;
				}
			}
		}
	}


	value = p_kernel_value;

	return 0;
};



int main(int argc, char** argv)
{

	//std::string prototxt_file_name = "D:/Caffe/models_out/tmp/deploy_viplfacenetNew3.prototxt";
	//std::string model_file_name = "D:/Caffe/models_out/tmp/viplfacenetNew3_256x256_lr0.07_M0.9_W0.0002_P0.5_WebFace_100K_iter_100000.caffemodel";
	//std::string label_file_name = "";// H: / WorkCode / SvnCode / CaffeNewest / data / ilsvrc12 / synset_words.txt";
	//std::vector<std::string > label_vector;
	//readLabel(label_file_name, label_vector);

	//std::string image_file_name = "D:/2.jpg";

	/*cv::Mat src1 = cv::imread(image_file_name);
	cv::Mat rszimage;
	cv::resize(src1, rszimage, cv::Size(80,80));
	cv::imwrite("D:/3.jpg", rszimage);*/

	std::string prototxt_file_name = "D:/workCode/TestModel/croplayer/fcn_3m.prototxt";
	std::string model_file_name = "D:/workCode/TestModel/croplayer/fcn_3m.caffemodel";
	//std::string inputfilename = "D:/workCode/TestModel/12MyModel.data";


	//std::string arv1 = "D:/workCode/TestModel/deploy.prototxt";
	//std::string arv2 = "D:/workCode/TestModel/fcn32s-heavy-pascal-final.caffemodel";

	caffe::Net<NetF>* _net = Net_Init_Load<NetF>(prototxt_file_name,model_file_name, caffe::TEST);

	caffe::MemoryDataLayer<NetF> *m_layer_ = (caffe::MemoryDataLayer<NetF> *)_net->layers()[0].get();
	m_layer_->set_batch_size(1);

	std::string feature_file_name = "D:/workCode/TestModel/croplayer/ip1.txt";
	std::string output_feature_name = "softmax_score";
	std::string image_file_name = "D:/workCode/TestModel/croplayer/test.png";
	PredictSingleFile(_net, image_file_name, cv::Size(240, 320), output_feature_name, feature_file_name);

	//system("pause");
	return 0;
}