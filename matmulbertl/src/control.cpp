#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "qop.h"

extern "C" {
v_arr
convWo(v_dt W) {
	#pragma HLS inline
	v_arr Wo;
	for(int i=0; i < VDATA_SIZE; i++)
		Wo.data[i] = W.data[i];
	return Wo;
}

void
controlsub(
	const v_dt *inW,            // Read-Only Weights
	hls::stream<v_arr> &o0_s,
	hls::stream<v_arr> &o1_s,
	hls::stream<v_arr> &o2_s,
	hls::stream<v_arr> &o3_s
)
{
	v_dt W;
	v_arr Wo;
	weightloop: for(int iter = 0; iter < (Tsize*Tsize*3)/(mems*VDATA_SIZE); iter++) {
		W = inW[iter];
		Wo = convWo(W);
		switch(iter%4) {
		case 0: o0_s.write(Wo); break;
		case 1: o1_s.write(Wo); break;
		case 2: o2_s.write(Wo); break;
		default: o3_s.write(Wo); break;
		}
	}
}


void
controlsubv(
	    const v_dt *in_tensor,           // Read-only tensor (1024, 14)
		hls::stream<v_arr> &ov00_s
		)
{
	v_dt W;
	for(int i=0; i < (Tsize*Veclen)/VDATA_SIZE; i++) {
		W = in_tensor[i];
		ov00_s.write(convWo(W));
	}
}

void
control(
	    const v_dt *in0,            // Read-Only Weights
	    const v_dt *in1,
	    const v_dt *in2,
	    const v_dt *in3,
	    const v_dt *in4,
	    const v_dt *in5,
	    const v_dt *in6,
	    const v_dt *in7,
	    const v_dt *in8,
	    const v_dt *in9,
	    const v_dt *in10,
	    const v_dt *in11,
	    const v_dt *in12,
	    const v_dt *in13,
	    const v_dt *in14,
	    const v_dt *in15,
	    const v_dt *in_tensor,           // Read-only tensor (1024, 14)
		hls::stream<v_arr> &o00_s,
		hls::stream<v_arr> &o01_s,
		hls::stream<v_arr> &o02_s,
		hls::stream<v_arr> &o03_s,
		hls::stream<v_arr> &o04_s,
		hls::stream<v_arr> &o05_s,
		hls::stream<v_arr> &o06_s,
		hls::stream<v_arr> &o07_s,
		hls::stream<v_arr> &o08_s,
		hls::stream<v_arr> &o09_s,
		hls::stream<v_arr> &o10_s,
		hls::stream<v_arr> &o11_s,
		hls::stream<v_arr> &o12_s,
		hls::stream<v_arr> &o13_s,
		hls::stream<v_arr> &o14_s,
		hls::stream<v_arr> &o15_s,
		hls::stream<v_arr> &o16_s,
		hls::stream<v_arr> &o17_s,
		hls::stream<v_arr> &o18_s,
		hls::stream<v_arr> &o19_s,
		hls::stream<v_arr> &o20_s,
		hls::stream<v_arr> &o21_s,
		hls::stream<v_arr> &o22_s,
		hls::stream<v_arr> &o23_s,
		hls::stream<v_arr> &o24_s,
		hls::stream<v_arr> &o25_s,
		hls::stream<v_arr> &o26_s,
		hls::stream<v_arr> &o27_s,
		hls::stream<v_arr> &o28_s,
		hls::stream<v_arr> &o29_s,
		hls::stream<v_arr> &o30_s,
		hls::stream<v_arr> &o31_s,
		hls::stream<v_arr> &o32_s,
		hls::stream<v_arr> &o33_s,
		hls::stream<v_arr> &o34_s,
		hls::stream<v_arr> &o35_s,
		hls::stream<v_arr> &o36_s,
		hls::stream<v_arr> &o37_s,
		hls::stream<v_arr> &o38_s,
		hls::stream<v_arr> &o39_s,
		hls::stream<v_arr> &o40_s,
		hls::stream<v_arr> &o41_s,
		hls::stream<v_arr> &o42_s,
		hls::stream<v_arr> &o43_s,
		hls::stream<v_arr> &o44_s,
		hls::stream<v_arr> &o45_s,
		hls::stream<v_arr> &o46_s,
		hls::stream<v_arr> &o47_s,
		hls::stream<v_arr> &o48_s,
		hls::stream<v_arr> &o49_s,
		hls::stream<v_arr> &o50_s,
		hls::stream<v_arr> &o51_s,
		hls::stream<v_arr> &o52_s,
		hls::stream<v_arr> &o53_s,
		hls::stream<v_arr> &o54_s,
		hls::stream<v_arr> &o55_s,
		hls::stream<v_arr> &o56_s,
		hls::stream<v_arr> &o57_s,
		hls::stream<v_arr> &o58_s,
		hls::stream<v_arr> &o59_s,
		hls::stream<v_arr> &o60_s,
		hls::stream<v_arr> &o61_s,
		hls::stream<v_arr> &o62_s,
		hls::stream<v_arr> &o63_s,
		hls::stream<v_arr> &ov00_s
	    )
{
	#pragma HLS INTERFACE m_axi port = in0 offset = slave bundle = gmem0 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem1 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem2 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in3 offset = slave bundle = gmem3 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in4 offset = slave bundle = gmem4 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in5 offset = slave bundle = gmem5 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in6 offset = slave bundle = gmem6 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in7 offset = slave bundle = gmem7 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in8 offset = slave bundle = gmem8 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in9 offset = slave bundle = gmem9 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in10 offset = slave bundle = gmem10 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in11 offset = slave bundle = gmem11 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in12 offset = slave bundle = gmem12 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in13 offset = slave bundle = gmem13 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in14 offset = slave bundle = gmem14 max_read_burst_length=128 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = in15 offset = slave bundle = gmem15 max_read_burst_length=32 max_write_burst_length=32
	#pragma HLS INTERFACE m_axi port = in_tensor offset = slave bundle = gmem16 

	#pragma HLS INTERFACE s_axilite port = in0
	#pragma HLS INTERFACE s_axilite port = in1
	#pragma HLS INTERFACE s_axilite port = in2
	#pragma HLS INTERFACE s_axilite port = in3
	#pragma HLS INTERFACE s_axilite port = in4
	#pragma HLS INTERFACE s_axilite port = in5
	#pragma HLS INTERFACE s_axilite port = in6
	#pragma HLS INTERFACE s_axilite port = in7
	#pragma HLS INTERFACE s_axilite port = in8
	#pragma HLS INTERFACE s_axilite port = in9
	#pragma HLS INTERFACE s_axilite port = in10
	#pragma HLS INTERFACE s_axilite port = in11
	#pragma HLS INTERFACE s_axilite port = in12
	#pragma HLS INTERFACE s_axilite port = in13
	#pragma HLS INTERFACE s_axilite port = in14
	#pragma HLS INTERFACE s_axilite port = in15
	#pragma HLS INTERFACE s_axilite port = in_tensor
	#pragma HLS INTERFACE s_axilite port = return
	#pragma HLS INTERFACE axis register_mode=both register port=o00_s
	#pragma HLS INTERFACE axis register_mode=both register port=o01_s
	#pragma HLS INTERFACE axis register_mode=both register port=o02_s
	#pragma HLS INTERFACE axis register_mode=both register port=o03_s
	#pragma HLS INTERFACE axis register_mode=both register port=o04_s
	#pragma HLS INTERFACE axis register_mode=both register port=o05_s
	#pragma HLS INTERFACE axis register_mode=both register port=o06_s
	#pragma HLS INTERFACE axis register_mode=both register port=o07_s
	#pragma HLS INTERFACE axis register_mode=both register port=o08_s
	#pragma HLS INTERFACE axis register_mode=both register port=o09_s
	#pragma HLS INTERFACE axis register_mode=both register port=o10_s
	#pragma HLS INTERFACE axis register_mode=both register port=o11_s
	#pragma HLS INTERFACE axis register_mode=both register port=o12_s
	#pragma HLS INTERFACE axis register_mode=both register port=o13_s
	#pragma HLS INTERFACE axis register_mode=both register port=o14_s
	#pragma HLS INTERFACE axis register_mode=both register port=o15_s
	#pragma HLS INTERFACE axis register_mode=both register port=o16_s
	#pragma HLS INTERFACE axis register_mode=both register port=o17_s
	#pragma HLS INTERFACE axis register_mode=both register port=o18_s
	#pragma HLS INTERFACE axis register_mode=both register port=o19_s
	#pragma HLS INTERFACE axis register_mode=both register port=o20_s
	#pragma HLS INTERFACE axis register_mode=both register port=o21_s
	#pragma HLS INTERFACE axis register_mode=both register port=o22_s
	#pragma HLS INTERFACE axis register_mode=both register port=o23_s
	#pragma HLS INTERFACE axis register_mode=both register port=o24_s
	#pragma HLS INTERFACE axis register_mode=both register port=o25_s
	#pragma HLS INTERFACE axis register_mode=both register port=o26_s
	#pragma HLS INTERFACE axis register_mode=both register port=o27_s
	#pragma HLS INTERFACE axis register_mode=both register port=o28_s
	#pragma HLS INTERFACE axis register_mode=both register port=o29_s
	#pragma HLS INTERFACE axis register_mode=both register port=o30_s
	#pragma HLS INTERFACE axis register_mode=both register port=o31_s
	#pragma HLS INTERFACE axis register_mode=both register port=o32_s
	#pragma HLS INTERFACE axis register_mode=both register port=o33_s
	#pragma HLS INTERFACE axis register_mode=both register port=o34_s
	#pragma HLS INTERFACE axis register_mode=both register port=o35_s
	#pragma HLS INTERFACE axis register_mode=both register port=o36_s
	#pragma HLS INTERFACE axis register_mode=both register port=o37_s
	#pragma HLS INTERFACE axis register_mode=both register port=o38_s
	#pragma HLS INTERFACE axis register_mode=both register port=o39_s
	#pragma HLS INTERFACE axis register_mode=both register port=o40_s
	#pragma HLS INTERFACE axis register_mode=both register port=o41_s
	#pragma HLS INTERFACE axis register_mode=both register port=o42_s
	#pragma HLS INTERFACE axis register_mode=both register port=o43_s
	#pragma HLS INTERFACE axis register_mode=both register port=o44_s
	#pragma HLS INTERFACE axis register_mode=both register port=o45_s
	#pragma HLS INTERFACE axis register_mode=both register port=o46_s
	#pragma HLS INTERFACE axis register_mode=both register port=o47_s
	#pragma HLS INTERFACE axis register_mode=both register port=o48_s
	#pragma HLS INTERFACE axis register_mode=both register port=o49_s
	#pragma HLS INTERFACE axis register_mode=both register port=o50_s
	#pragma HLS INTERFACE axis register_mode=both register port=o51_s
	#pragma HLS INTERFACE axis register_mode=both register port=o52_s
	#pragma HLS INTERFACE axis register_mode=both register port=o53_s
	#pragma HLS INTERFACE axis register_mode=both register port=o54_s
	#pragma HLS INTERFACE axis register_mode=both register port=o55_s
	#pragma HLS INTERFACE axis register_mode=both register port=o56_s
	#pragma HLS INTERFACE axis register_mode=both register port=o57_s
	#pragma HLS INTERFACE axis register_mode=both register port=o58_s
	#pragma HLS INTERFACE axis register_mode=both register port=o59_s
	#pragma HLS INTERFACE axis register_mode=both register port=o60_s
	#pragma HLS INTERFACE axis register_mode=both register port=o61_s
	#pragma HLS INTERFACE axis register_mode=both register port=o62_s
	#pragma HLS INTERFACE axis register_mode=both register port=o63_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov00_s
	#pragma HLS dataflow
	controlsub(in0,  o00_s, o01_s, o02_s, o03_s);
	controlsub(in1,  o04_s, o05_s, o06_s, o07_s);
	controlsub(in2,  o08_s, o09_s, o10_s, o11_s);
	controlsub(in3,  o12_s, o13_s, o14_s, o15_s);
	controlsub(in4,  o16_s, o17_s, o18_s, o19_s);
	controlsub(in5,  o20_s, o21_s, o22_s, o23_s);
	controlsub(in6,  o24_s, o25_s, o26_s, o27_s);
	controlsub(in7,  o28_s, o29_s, o30_s, o31_s);
	controlsub(in8,  o32_s, o33_s, o34_s, o35_s);
	controlsub(in9,  o36_s, o37_s, o38_s, o39_s);
	controlsub(in10, o40_s, o41_s, o42_s, o43_s);
	controlsub(in11, o44_s, o45_s, o46_s, o47_s);
	controlsub(in12, o48_s, o49_s, o50_s, o51_s);
	controlsub(in13, o52_s, o53_s, o54_s, o55_s);
	controlsub(in14, o56_s, o57_s, o58_s, o59_s);
	controlsub(in15, o60_s, o61_s, o62_s, o63_s);
	controlsubv(in_tensor, ov00_s);
}
}
