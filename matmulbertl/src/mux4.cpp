#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "qop.h"

extern "C" {

void
mux4(
	hls::stream<v_arr> &i_s,
	hls::stream<v_arr> &o1_s,
	hls::stream<v_arr> &o2_s,
	hls::stream<v_arr> &o3_s,
	hls::stream<v_arr> &o4_s
)
{
	#pragma HLS INTERFACE axis register_mode=both register port=i_s
	#pragma HLS INTERFACE axis register_mode=none register port=o1_s
	#pragma HLS INTERFACE axis register_mode=none register port=o2_s
	#pragma HLS INTERFACE axis register_mode=none register port=o3_s
	#pragma HLS INTERFACE axis register_mode=none register port=o4_s
	#pragma HLS interface ap_ctrl_none port=return
	
	v_arr v;
	i_s.read(v);
	o1_s.write(v);
	i_s.read(v);
	o1_s.write(v);
	i_s.read(v);
	o2_s.write(v);
	i_s.read(v);
	o2_s.write(v);
	i_s.read(v);
	o3_s.write(v);
	i_s.read(v);
	o3_s.write(v);
	i_s.read(v);
	o4_s.write(v);
	i_s.read(v);
	o4_s.write(v);
}
}
