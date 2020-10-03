#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "krnl_matmul.h"

enum {
	mems = 16,
	cores = 64,
	Tp2 = 10,
	Tsize = 1<<Tp2,             /* 1024 */
	Msize = (Tsize/cores),      /* 16 */
	Vsize = Msize/VDATA_SIZE,   /* 1 */
	Vshift = 2,
	Veclen = 14,
};

extern "C" {

typedef struct v_inWrrtype{
	Dt data[Msize];
} v_arr;

void
qop(
	hls::stream<v_arr> &inW_s,
	hls::stream<v_arr> &inV_s,
	hls::stream<v_dt> &o_s
)
{
#pragma HLS INTERFACE axis register_mode=both register port=inW_s
#pragma HLS INTERFACE axis port=inV_s
#pragma HLS INTERFACE axis register_mode=both register port=ov_s
	Dt mulres[Vsize*VDATA_SIZE];
	Dt addres1[8];
	Dt addres2[4];
	Dt addres3[2];

	v_arr inW;
	v_arr inV[Veclen];
	Dt res;
	v_dt o;

	#pragma HLS array_partition variable = mulres dim=0
	#pragma HLS array_partition variable = addres1 dim=0
	#pragma HLS array_partition variable = addres2 dim=0
	#pragma HLS array_partition variable = addres3 dim=0

	#pragma HLS interface ap_ctrl_none port=return
	#pragma HLS allocation instances=fmul limit=16 operation
	#pragma HLS allocation instances=fadd limit=15 operation
	#pragma HLS allocation instances=mul limit=16 operation

	for(int v = 0; v < Veclen; v++) {
		#pragma HLS pipeline II=1
		inV_s.read(inV[v]);
	}

	for(int iter = 0; iter < (Tsize*Tsize*3)/(mems*VDATA_SIZE); iter++) {
		#pragma HLS pipeline II=14
		inW_s.read(inW);
		for(int v = 0; v < Veclen; v++) {
			l3:for(int k = 0 ; k < Msize; k++) {
				mulres[k] = inW.data[k] * inV[v].data[k];
			}

			a1: for(int i = 0; i < Vsize*VDATA_SIZE/2; i++) {
				addres1[i] = mulres[i*2] + mulres[i*2+1];
			}

			a2: for(int i = 0; i < Vsize*VDATA_SIZE/4; i++) {
				addres2[i] = addres1[i*2] + addres1[i*2+1];
			}

			a3: for(int i = 0; i < Vsize*VDATA_SIZE/8; i++) {
				addres3[i] = addres2[i*2] + addres2[i*2+1];
			}

			res = addres3[0] + addres3[1];
			o.data[v] = res;
		}
		o_s.write(o);
	}
}

void
addt16(
	hls::stream<Dt> &in0_s,
	hls::stream<Dt> &in1_s,
	hls::stream<Dt> &in2_s,
	hls::stream<Dt> &in3_s,
	hls::stream<Dt> &in4_s,
	hls::stream<Dt> &in5_s,
	hls::stream<Dt> &in6_s,
	hls::stream<Dt> &in7_s,
	hls::stream<Dt> &in8_s,
	hls::stream<Dt> &in9_s,
	hls::stream<Dt> &in10_s,
	hls::stream<Dt> &in11_s,
	hls::stream<Dt> &in12_s,
	hls::stream<Dt> &in13_s,
	hls::stream<Dt> &in14_s,
	hls::stream<Dt> &in15_s,
	hls::stream<Dt> &o_s
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
	#pragma HLS INTERFACE axis register_mode=both register port=in8_s
	#pragma HLS INTERFACE axis register_mode=both register port=in9_s
	#pragma HLS INTERFACE axis register_mode=both register port=in10_s
	#pragma HLS INTERFACE axis register_mode=both register port=in11_s
	#pragma HLS INTERFACE axis register_mode=both register port=in12_s
	#pragma HLS INTERFACE axis register_mode=both register port=in13_s
	#pragma HLS INTERFACE axis register_mode=both register port=in14_s
	#pragma HLS INTERFACE axis register_mode=both register port=in15_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov_s
	Dt in[16];
	Dt imm[3][8];
	Dt o;

	#pragma HLS array_partition variable = in dim=0
	#pragma HLS array_partition variable = imm dim=0
	#pragma HLS pipeline II=1
	#pragma HLS interface ap_ctrl_none port=return
	#pragma HLS allocation instances=fadd limit=15 operation

	in0_s.read(in[0]);
	in1_s.read(in[1]);
	in2_s.read(in[2]);
	in3_s.read(in[3]);
	in4_s.read(in[4]);
	in5_s.read(in[5]);
	in6_s.read(in[6]);
	in7_s.read(in[7]);
	in8_s.read(in[8]);
	in9_s.read(in[9]);
	in10_s.read(in[10]);
	in11_s.read(in[11]);
	in12_s.read(in[12]);
	in13_s.read(in[13]);
	in14_s.read(in[14]);
	in15_s.read(in[15]);

	a0: for(int i = 0; i < 8; i++) {
		imm[0][i] = in[i*2] + in[i*2+1];
	}

	a1: for(int i = 0; i < 4; i++) {
		imm[1][i] = imm[0][i*2] + imm[0][i*2+1];
	}

	a2: for(int i = 0; i < 2; i++) {
		imm[2][i] = imm[2][i*2] + imm[2][i*2+1];
	}

	o = imm[2][0] + imm[2][1];
	o_s.write(o);
}

void
addt4(
	hls::stream<Dt> &in0_s,
	hls::stream<Dt> &in1_s,
	hls::stream<Dt> &in2_s,
	hls::stream<Dt> &in3_s,
	hls::stream<Dt> &o_s
)
{
	#pragma HLS INTERFACE axis register_mode=both register port=in0_s
	#pragma HLS INTERFACE axis register_mode=both register port=in1_s
	#pragma HLS INTERFACE axis register_mode=both register port=in2_s
	#pragma HLS INTERFACE axis register_mode=both register port=in3_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov_s
	Dt in[4];
	Dt imm[1][2];
	Dt o;

	#pragma HLS array_partition variable = in dim=0
	#pragma HLS array_partition variable = imm dim=0
	#pragma HLS pipeline II=1
	#pragma HLS interface ap_ctrl_none port=return
	#pragma HLS allocation instances=fadd limit=3 operation

	in0_s.read(in[0]);
	in1_s.read(in[1]);
	in2_s.read(in[2]);
	in3_s.read(in[3]);

	a0: for(int i = 0; i < 2; i++) {
		imm[0][i] = in[i*2] + in[i*2+1];
	}

	o = imm[0][0] + imm[0][1];
	o_s.write(o);
}

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

int base;
void
controlsubvv (
	v_dt *in_tensor,           // Read-only tensor (1024, 14)
	hls::stream<v_arr> &ov00_s
	)
{
	#pragma HLS inline
	v_dt W;
	for(int i=0; i < Veclen; i++) {
		W = in_tensor[base+i];
		ov00_s.write(convWo(W));
		base += Veclen;
	}
}

void
controlsubv(
	    v_dt *in_tensor,           // Read-only tensor (1024, 14)
		hls::stream<v_arr> &ov00_s,
		hls::stream<v_arr> &ov01_s,
		hls::stream<v_arr> &ov02_s,
		hls::stream<v_arr> &ov03_s,
		hls::stream<v_arr> &ov04_s,
		hls::stream<v_arr> &ov05_s,
		hls::stream<v_arr> &ov06_s,
		hls::stream<v_arr> &ov07_s,
		hls::stream<v_arr> &ov08_s,
		hls::stream<v_arr> &ov09_s,
		hls::stream<v_arr> &ov10_s,
		hls::stream<v_arr> &ov11_s,
		hls::stream<v_arr> &ov12_s,
		hls::stream<v_arr> &ov13_s,
		hls::stream<v_arr> &ov14_s,
		hls::stream<v_arr> &ov15_s,
		hls::stream<v_arr> &ov16_s,
		hls::stream<v_arr> &ov17_s,
		hls::stream<v_arr> &ov18_s,
		hls::stream<v_arr> &ov19_s,
		hls::stream<v_arr> &ov20_s,
		hls::stream<v_arr> &ov21_s,
		hls::stream<v_arr> &ov22_s,
		hls::stream<v_arr> &ov23_s,
		hls::stream<v_arr> &ov24_s,
		hls::stream<v_arr> &ov25_s,
		hls::stream<v_arr> &ov26_s,
		hls::stream<v_arr> &ov27_s,
		hls::stream<v_arr> &ov28_s,
		hls::stream<v_arr> &ov29_s,
		hls::stream<v_arr> &ov30_s,
		hls::stream<v_arr> &ov31_s,
		hls::stream<v_arr> &ov32_s,
		hls::stream<v_arr> &ov33_s,
		hls::stream<v_arr> &ov34_s,
		hls::stream<v_arr> &ov35_s,
		hls::stream<v_arr> &ov36_s,
		hls::stream<v_arr> &ov37_s,
		hls::stream<v_arr> &ov38_s,
		hls::stream<v_arr> &ov39_s,
		hls::stream<v_arr> &ov40_s,
		hls::stream<v_arr> &ov41_s,
		hls::stream<v_arr> &ov42_s,
		hls::stream<v_arr> &ov43_s,
		hls::stream<v_arr> &ov44_s,
		hls::stream<v_arr> &ov45_s,
		hls::stream<v_arr> &ov46_s,
		hls::stream<v_arr> &ov47_s,
		hls::stream<v_arr> &ov48_s,
		hls::stream<v_arr> &ov49_s,
		hls::stream<v_arr> &ov50_s,
		hls::stream<v_arr> &ov51_s,
		hls::stream<v_arr> &ov52_s,
		hls::stream<v_arr> &ov53_s,
		hls::stream<v_arr> &ov54_s,
		hls::stream<v_arr> &ov55_s,
		hls::stream<v_arr> &ov56_s,
		hls::stream<v_arr> &ov57_s,
		hls::stream<v_arr> &ov58_s,
		hls::stream<v_arr> &ov59_s,
		hls::stream<v_arr> &ov60_s,
		hls::stream<v_arr> &ov61_s,
		hls::stream<v_arr> &ov62_s,
		hls::stream<v_arr> &ov63_s
		)
{
	base = 0;
	controlsubvv(in_tensor, ov00_s);
	controlsubvv(in_tensor, ov01_s);
	controlsubvv(in_tensor, ov02_s);
	controlsubvv(in_tensor, ov03_s);
	controlsubvv(in_tensor, ov04_s);
	controlsubvv(in_tensor, ov05_s);
	controlsubvv(in_tensor, ov06_s);
	controlsubvv(in_tensor, ov07_s);
	controlsubvv(in_tensor, ov08_s);
	controlsubvv(in_tensor, ov09_s);
	controlsubvv(in_tensor, ov10_s);
	controlsubvv(in_tensor, ov11_s);
	controlsubvv(in_tensor, ov12_s);
	controlsubvv(in_tensor, ov13_s);
	controlsubvv(in_tensor, ov14_s);
	controlsubvv(in_tensor, ov15_s);
	controlsubvv(in_tensor, ov16_s);
	controlsubvv(in_tensor, ov17_s);
	controlsubvv(in_tensor, ov18_s);
	controlsubvv(in_tensor, ov19_s);
	controlsubvv(in_tensor, ov20_s);
	controlsubvv(in_tensor, ov21_s);
	controlsubvv(in_tensor, ov22_s);
	controlsubvv(in_tensor, ov23_s);
	controlsubvv(in_tensor, ov24_s);
	controlsubvv(in_tensor, ov25_s);
	controlsubvv(in_tensor, ov26_s);
	controlsubvv(in_tensor, ov27_s);
	controlsubvv(in_tensor, ov28_s);
	controlsubvv(in_tensor, ov29_s);
	controlsubvv(in_tensor, ov30_s);
	controlsubvv(in_tensor, ov31_s);
	controlsubvv(in_tensor, ov32_s);
	controlsubvv(in_tensor, ov33_s);
	controlsubvv(in_tensor, ov34_s);
	controlsubvv(in_tensor, ov35_s);
	controlsubvv(in_tensor, ov36_s);
	controlsubvv(in_tensor, ov37_s);
	controlsubvv(in_tensor, ov38_s);
	controlsubvv(in_tensor, ov39_s);
	controlsubvv(in_tensor, ov40_s);
	controlsubvv(in_tensor, ov41_s);
	controlsubvv(in_tensor, ov42_s);
	controlsubvv(in_tensor, ov43_s);
	controlsubvv(in_tensor, ov44_s);
	controlsubvv(in_tensor, ov45_s);
	controlsubvv(in_tensor, ov46_s);
	controlsubvv(in_tensor, ov47_s);
	controlsubvv(in_tensor, ov48_s);
	controlsubvv(in_tensor, ov49_s);
	controlsubvv(in_tensor, ov50_s);
	controlsubvv(in_tensor, ov51_s);
	controlsubvv(in_tensor, ov52_s);
	controlsubvv(in_tensor, ov53_s);
	controlsubvv(in_tensor, ov54_s);
	controlsubvv(in_tensor, ov55_s);
	controlsubvv(in_tensor, ov56_s);
	controlsubvv(in_tensor, ov57_s);
	controlsubvv(in_tensor, ov58_s);
	controlsubvv(in_tensor, ov59_s);
	controlsubvv(in_tensor, ov60_s);
	controlsubvv(in_tensor, ov61_s);
	controlsubvv(in_tensor, ov62_s);
	controlsubvv(in_tensor, ov63_s);
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
	    v_dt *in_tensor,           // Read-only tensor (64, 14)
		hls::stream<v_arr> &ov00_s,
		hls::stream<v_arr> &ov01_s,
		hls::stream<v_arr> &ov02_s,
		hls::stream<v_arr> &ov03_s,
		hls::stream<v_arr> &ov04_s,
		hls::stream<v_arr> &ov05_s,
		hls::stream<v_arr> &ov06_s,
		hls::stream<v_arr> &ov07_s,
		hls::stream<v_arr> &ov08_s,
		hls::stream<v_arr> &ov09_s,
		hls::stream<v_arr> &ov10_s,
		hls::stream<v_arr> &ov11_s,
		hls::stream<v_arr> &ov12_s,
		hls::stream<v_arr> &ov13_s,
		hls::stream<v_arr> &ov14_s,
		hls::stream<v_arr> &ov15_s,
		hls::stream<v_arr> &ov16_s,
		hls::stream<v_arr> &ov17_s,
		hls::stream<v_arr> &ov18_s,
		hls::stream<v_arr> &ov19_s,
		hls::stream<v_arr> &ov20_s,
		hls::stream<v_arr> &ov21_s,
		hls::stream<v_arr> &ov22_s,
		hls::stream<v_arr> &ov23_s,
		hls::stream<v_arr> &ov24_s,
		hls::stream<v_arr> &ov25_s,
		hls::stream<v_arr> &ov26_s,
		hls::stream<v_arr> &ov27_s,
		hls::stream<v_arr> &ov28_s,
		hls::stream<v_arr> &ov29_s,
		hls::stream<v_arr> &ov30_s,
		hls::stream<v_arr> &ov31_s,
		hls::stream<v_arr> &ov32_s,
		hls::stream<v_arr> &ov33_s,
		hls::stream<v_arr> &ov34_s,
		hls::stream<v_arr> &ov35_s,
		hls::stream<v_arr> &ov36_s,
		hls::stream<v_arr> &ov37_s,
		hls::stream<v_arr> &ov38_s,
		hls::stream<v_arr> &ov39_s,
		hls::stream<v_arr> &ov40_s,
		hls::stream<v_arr> &ov41_s,
		hls::stream<v_arr> &ov42_s,
		hls::stream<v_arr> &ov43_s,
		hls::stream<v_arr> &ov44_s,
		hls::stream<v_arr> &ov45_s,
		hls::stream<v_arr> &ov46_s,
		hls::stream<v_arr> &ov47_s,
		hls::stream<v_arr> &ov48_s,
		hls::stream<v_arr> &ov49_s,
		hls::stream<v_arr> &ov50_s,
		hls::stream<v_arr> &ov51_s,
		hls::stream<v_arr> &ov52_s,
		hls::stream<v_arr> &ov53_s,
		hls::stream<v_arr> &ov54_s,
		hls::stream<v_arr> &ov55_s,
		hls::stream<v_arr> &ov56_s,
		hls::stream<v_arr> &ov57_s,
		hls::stream<v_arr> &ov58_s,
		hls::stream<v_arr> &ov59_s,
		hls::stream<v_arr> &ov60_s,
		hls::stream<v_arr> &ov61_s,
		hls::stream<v_arr> &ov62_s,
		hls::stream<v_arr> &ov63_s
	    )
{
	#pragma HLS INTERFACE m_axi port = in0 offset = slave bundle = gmem0
	#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem1
	#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem2
	#pragma HLS INTERFACE m_axi port = in3 offset = slave bundle = gmem3
	#pragma HLS INTERFACE m_axi port = in4 offset = slave bundle = gmem4
	#pragma HLS INTERFACE m_axi port = in5 offset = slave bundle = gmem5
	#pragma HLS INTERFACE m_axi port = in6 offset = slave bundle = gmem6
	#pragma HLS INTERFACE m_axi port = in7 offset = slave bundle = gmem7
	#pragma HLS INTERFACE m_axi port = in8 offset = slave bundle = gmem8
	#pragma HLS INTERFACE m_axi port = in9 offset = slave bundle = gmem9
	#pragma HLS INTERFACE m_axi port = in10 offset = slave bundle = gmem10
	#pragma HLS INTERFACE m_axi port = in11 offset = slave bundle = gmem11
	#pragma HLS INTERFACE m_axi port = in12 offset = slave bundle = gmem12
	#pragma HLS INTERFACE m_axi port = in13 offset = slave bundle = gmem13
	#pragma HLS INTERFACE m_axi port = in14 offset = slave bundle = gmem14
	#pragma HLS INTERFACE m_axi port = in15 offset = slave bundle = gmem15
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
	#pragma HLS INTERFACE axis register_mode=both register port=ov01_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov02_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov03_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov04_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov05_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov06_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov07_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov08_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov09_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov10_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov11_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov12_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov13_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov14_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov15_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov16_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov17_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov18_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov19_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov20_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov21_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov22_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov23_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov24_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov25_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov26_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov27_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov28_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov29_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov30_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov31_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov32_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov33_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov34_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov35_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov36_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov37_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov38_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov39_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov40_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov41_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov42_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov43_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov44_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov45_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov46_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov47_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov48_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov49_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov50_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov51_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov52_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov53_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov54_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov55_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov56_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov57_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov58_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov59_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov60_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov61_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov62_s
	#pragma HLS INTERFACE axis register_mode=both register port=ov63_s
	#pragma HLS INTERFACE s_axilite port = in_tensor
	#pragma HLS INTERFACE s_axilite port = return
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
	controlsubv(in_tensor,
			ov00_s, ov01_s, ov02_s, ov03_s, ov04_s, ov05_s, ov06_s, ov07_s, ov08_s, ov09_s,
			ov10_s, ov11_s, ov12_s, ov13_s, ov14_s, ov15_s, ov16_s, ov17_s, ov18_s, ov19_s,
			ov20_s, ov21_s, ov22_s, ov23_s, ov24_s, ov25_s, ov26_s, ov27_s, ov28_s, ov29_s,
			ov30_s, ov31_s, ov32_s, ov33_s, ov34_s, ov35_s, ov36_s, ov37_s, ov38_s, ov39_s,
			ov40_s, ov41_s, ov42_s, ov43_s, ov44_s, ov45_s, ov46_s, ov47_s, ov48_s, ov49_s,
			ov50_s, ov51_s, ov52_s, ov53_s, ov54_s, ov55_s, ov56_s, ov57_s, ov58_s, ov59_s,
			ov60_s, ov61_s, ov62_s, ov63_s);
}
}
