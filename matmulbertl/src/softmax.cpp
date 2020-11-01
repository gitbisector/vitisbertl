#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "qop.h"
#include <hls_math.h>

extern "C" {
void
softmax(
    v_dt *i_tensor,
    v_dt *o_tensor
    )
{
	#pragma HLS INTERFACE m_axi port=i_tensor offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=o_tensor offset=slave bundle=gmem1
	#pragma HLS INTERFACE s_axilite bundle = control port = i_tensor
	#pragma HLS INTERFACE s_axilite bundle = control port = o_tensor
	#pragma HLS INTERFACE s_axilite bundle = control port = return
	#pragma HLS allocation instances=fadd limit=64 operation
	v_dt v, o;

	#pragma HLS array_partition variable=o dim=1
	#pragma HLS array_partition variable=v dim=1

	Dt m;
	for(int i = 0; i < Tsize/VDATA_SIZE; i++) {
		v = i_tensor[i];
		for(int d = 0; d < VDATA_SIZE; d++) {
			#pragma HLS pipeline II=1
			Dt n = v.data[d];
			if((i==0 && d==0) || n > m)
				m = n;
		}
	}

	It sum = 0;
	It lsum;
	for(int i = 0; i < Tsize/VDATA_SIZE; i++) {
		v = i_tensor[i];
		for(int d = 0; d < VDATA_SIZE; d++) {
			#pragma HLS pipeline II=1
			sum += hls::exp(v.data[d] - m);
		}
	}

	lsum = hls::log(sum);
	for(int i = 0; i < Tsize/VDATA_SIZE; i++) {
		#pragma HLS pipeline II=1
		v = i_tensor[i];
		for(int d = 0; d < VDATA_SIZE; d++) {
			o.data[d] = hls::exp( v.data[d] - m - lsum);
		}
		o_tensor[i] = o;
	}
}
}
