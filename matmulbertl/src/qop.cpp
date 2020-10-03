#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "qop.h"

extern "C" {

void
qop(
	hls::stream<v_arr> &inW_s0,
	hls::stream<v_arr> &inV_s,
	hls::stream<v_arr> &inW_s1,
	hls::stream<It> &o_s
)
{
#pragma HLS INTERFACE axis register_mode=both register port=inW_s0
#pragma HLS INTERFACE axis port=inV_s
#pragma HLS INTERFACE axis register_mode=both register port=o_s
#pragma HLS INTERFACE axis register_mode=both register port=inW_s1

#pragma HLS interface ap_ctrl_none port=return

	v_arr inW[2];
	v_arr inV[Veclen*2][2];
	It res;
	int vofs;

	#pragma HLS resource core=RAM_S2P_BRAM variable=inV
	#pragma HLS array_partition variable=inW dim=1
	#pragma HLS array_partition variable=inV dim=2
	#pragma HLS array_partition variable=inV dim=3
	#pragma HLS allocation instances=fmul limit=32 operation
	#pragma HLS allocation instances=fadd limit=31 operation
	#pragma HLS allocation instances=mul  limit=32 operation
	#pragma HLS allocation instances=hmul limit=32 operation
	#pragma HLS allocation instances=hadd limit=31 operation

	for(int v = 0; v < Veclen*(Msize/VDATA_SIZE); v++) {
		#pragma HLS pipeline II=1
		inV_s.read(inV[v][0]);
		inV_s.read(inV[v][1]);
	}

	weightloop: for(int iter = 0; iter < (Tsize*Tsize*3)/(cores*VDATA_SIZE); iter++) {
		#pragma HLS pipeline II=14
		inW_s0.read(inW[0]);
		inW_s1.read(inW[1]);
		vofs = Veclen * (iter%(Msize/VDATA_SIZE));
		for(int v = 0; v < Veclen; v++) {
			l3:for(int k = 0 ; k < VDATA_SIZE; k++) {
				res = ((k==0)?(It)0:res)+ (inW[0].data[k] * inV[v+vofs][0].data[k]) + (inW[1].data[k] * inV[v+vofs][1].data[k]);
			}
			o_s.write(res);
		}
	}
}
}
