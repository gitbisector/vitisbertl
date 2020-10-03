#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "qop.h"

extern "C" {

void
addt16(
	hls::stream<It> &in0_s,
	hls::stream<It> &in1_s,
	hls::stream<It> &in2_s,
	hls::stream<It> &in3_s,
	hls::stream<It> &in4_s,
	hls::stream<It> &in5_s,
	hls::stream<It> &in6_s,
	hls::stream<It> &in7_s,
	hls::stream<It> &o_s
)
{
	#pragma HLS INTERFACE axis register_mode=both register port=in0_s
	#pragma HLS INTERFACE axis register_mode=both register port=in1_s
	#pragma HLS INTERFACE axis register_mode=both register port=in2_s
	#pragma HLS INTERFACE axis register_mode=both register port=in3_s
	#pragma HLS INTERFACE axis register_mode=both register port=in4_s
	#pragma HLS INTERFACE axis register_mode=both register port=in5_s
	#pragma HLS INTERFACE axis register_mode=both register port=in6_s
	#pragma HLS INTERFACE axis register_mode=both register port=in7_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov_s
	It in[8];
	It imm[2][4];
	It o;

	#pragma HLS array_partition variable=in  dim=0
	#pragma HLS array_partition variable=imm dim=0
	#pragma HLS pipeline II=1
	#pragma HLS interface ap_ctrl_none port=return
	#pragma HLS allocation instances=fadd limit=15 operation
	#pragma HLS allocation instances=hadd limit=15 operation

	in0_s.read(in[0]);
	in1_s.read(in[1]);
	in2_s.read(in[2]);
	in3_s.read(in[3]);
	in4_s.read(in[4]);
	in5_s.read(in[5]);
	in6_s.read(in[6]);
	in7_s.read(in[7]);

	i0: for(int i = 0; i < 4; i++) {
		imm[0][i] = in[i*2] + in[i*2+1];
	}

	a0: for(int i = 0; i < 2; i++) {
		imm[1][i] = imm[0][i*2] + imm[0][i*2+1];
	}

	o = imm[1][0] + imm[1][1];
	o_s.write(o);
}
}
