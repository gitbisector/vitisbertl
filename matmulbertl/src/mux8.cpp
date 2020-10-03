#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "qop.h"

extern "C" {

void
mux8(
	hls::stream<v_arr> &i_s,
	hls::stream<v_arr> &o1_s,
	hls::stream<v_arr> &o2_s,
	hls::stream<v_arr> &o3_s,
	hls::stream<v_arr> &o4_s,
	hls::stream<v_arr> &o5_s,
	hls::stream<v_arr> &o6_s,
	hls::stream<v_arr> &o7_s,
	hls::stream<v_arr> &o8_s
)
{
	#pragma HLS INTERFACE axis register_mode=both register port=i_s
	#pragma HLS INTERFACE axis register_mode=none register port=o1_s
	#pragma HLS INTERFACE axis register_mode=none register port=o2_s
	#pragma HLS INTERFACE axis register_mode=none register port=o3_s
	#pragma HLS INTERFACE axis register_mode=none register port=o4_s
	#pragma HLS INTERFACE axis register_mode=none register port=o5_s
	#pragma HLS INTERFACE axis register_mode=none register port=o6_s
	#pragma HLS INTERFACE axis register_mode=none register port=o7_s
	#pragma HLS INTERFACE axis register_mode=none register port=o8_s
	#pragma HLS interface ap_ctrl_none port=return

	v_arr v;
	for(int i =0; i < 8; i++) {
		i_s.read(v);
		o1_s.write(v);
	}
	for(int i =0; i < 8; i++) {
		i_s.read(v);
		o2_s.write(v);
	}
	for(int i =0; i < 8; i++) {
		i_s.read(v);
		o3_s.write(v);
	}
	for(int i =0; i < 8; i++) {
		i_s.read(v);
		o4_s.write(v);
	}
	for(int i =0; i < 8; i++) {
		i_s.read(v);
		o5_s.write(v);
	}
	for(int i =0; i < 8; i++) {
		i_s.read(v);
		o6_s.write(v);
	}
	for(int i =0; i < 8; i++) {
		i_s.read(v);
		o7_s.write(v);
	}
	for(int i =0; i < 8; i++) {
		i_s.read(v);
		o8_s.write(v);
	}
}
}
