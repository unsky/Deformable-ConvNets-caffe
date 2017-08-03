#include <vector>
#include <iostream>

#include "caffe/layers/deformable_conv_layer.hpp"
#include "caffe/util/deformable_im2col.hpp"
using namespace std;
namespace caffe {
template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* weights = this->blobs_[0]->gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* offset = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    cout<<endl<<"debug:"<<endl;
    Dtype* aa =top[0]->mutable_cpu_data();
    Dtype* bb =bottom[0]->mutable_cpu_data();
    Dtype* cc =bottom[0]->mutable_cpu_data();
   cout << top[0]->shape_string() << endl;
  cout << aa[0] + aa[1] << endl;
    cout << bb[0] + bb[1] << endl;
      cout << cc[0] + cc[1] << endl; 
  
    cout << bottom[0]->asum_data() << endl; 
    cout << bottom[1]->asum_data() << endl;   
    cout << top[0]->asum_data() << endl; 

    const int* kernel_shape_data = this->kernel_shape_.cpu_data();
    const int* pad_data = this->pad_.cpu_data();
    const int* stride_data = this->stride_.cpu_data();
    const int* dilation_data = this->dilation_.cpu_data();
    const int height = bottom[0]->shape(1);
    const int width = bottom[0]->shape(0);
    const int channels = this->channels_;
    const uint32_t c = 4;

    Dtype* col_buff = bottom[0]->mutable_gpu_data();


    for (int n = 0; n < this->num_; ++n) {
            deformable_im2col_gpu(bottom_data + n*this->bottom_dim_, //data_col
            offset + n*this->bottom_dim_,//offset
            channels,
            height,//height 
            width,//width
            kernel_shape_data[0],//
            kernel_shape_data[1],
            pad_data[0],
            pad_data[1],
            stride_data[0],
            stride_data[1],
            dilation_data[0],
            dilation_data[1],
            c,
            col_buff + n*this->bottom_dim_);

for (int g = 0; g < this->group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->conv_out_channels_ /
        this->group_, this->conv_out_spatial_dim_, this->kernel_dim_,
        (Dtype)1., weights + this->weight_offset_ * g, col_buff + this->col_offset_ * g,
        (Dtype)0., top_data + n * this->top_dim_ + this->output_offset_ * g);
      }
      }


cout << top[0]->shape_string() << endl; 
cout << top[0]->asum_data() << endl; 
cout<<"aaa"<<endl;
cout << bottom[1]->asum_data() << endl; 
cout << bottom[0]->asum_data() << endl; 
cout << bottom[1]->shape_string() << endl;    
cout << top[0]->shape_string() << endl; 
cout << "...................."<<endl;
}

template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      Dtype * data1 = bottom[0]->mutable_cpu_data();
      Dtype * data2 = bottom[1]->mutable_cpu_data();
      for(int i = 0; i < bottom[0]->count(); i++)
         data1[i] = 0;

      for(int i = 0; i < bottom[1]->count(); i++)
         data2[i] = 0;   
        
}
//


INSTANTIATE_LAYER_GPU_FUNCS(DeformableConvolutionLayer);


}  // namespace caffe
