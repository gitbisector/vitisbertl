#include "ap_axi_sdata.h"
#define AP_INT_MAX_W 2048
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "qop.h"
#include <iostream>

typedef struct v_in1024 { Dt data[Tsize]; } v_arr1024;
typedef struct v_in128 { Dt data[Tsize/8]; } v_arr128;


void
onemath(Dt inV[Tsize], Dt W[Tsize], hls::stream<ap_axiu<Itsize,0,0,0> > &o_s)
{
	It imm[MopersP2+1+Qop_pathsP2][Mopers*Qop_paths];
	#pragma HLS pipeline II=1 enable_flush
	#pragma HLS array_partition variable=imm dim=0
//	#pragma HLS allocation instances=mul  limit=1024 operation
	l1: for(int b = 0; b < Tsize; b++) {
		imm[0][b] = (Dt)(inV[b] * W[b]);
	}
	for(int s = 0; s < MopersP2+Qop_pathsP2; s++) {
		for(int k = 0; k < (Tsize>>(s+1)); k++) {
			imm[s+1][k] = imm[s][k] + imm[s][k+(Tsize>>(s+1))];
		}
	}
	ap_axiu<Itsize,0,0,0> o_s_v;
	o_s_v.data(imm[0][0].width-1, 0) = imm[MopersP2+Qop_pathsP2][0](imm[0][0].width-1, 0);
	o_s.write(o_s_v);
}

void
fetchV(
		hls::stream<v_arr1024 > &inV_s0,
		Dt V[Tsize]
	)
{
	v_arr1024 inV;
	#pragma HLS array_partition variable=inV dim=1
	#pragma HLS pipeline II=1
	inV_s0.read(inV);
	for(int i = 0; i < Tsize; i++) {
		V[i] = inV.data[i];
	}
}

void
fetchW(
		hls::stream<v_arr1024 > &inW_s0,
		Dt W[Tsize]
	)
{
	v_arr1024 inW;
	#pragma HLS array_partition variable=inW dim=1
	#pragma HLS pipeline II=1
	inW_s0.read(inW);
	for(int i = 0; i < Tsize; i++) {
			W[i].range() = inW.data[i];
	}
}

/* Collect 1024 */
template <int lalala> void
makeone(v_arr v[8],	v_arr128 &oW)
{
	#pragma HLS pipeline II=1
	for(int i = 0; i < 8; i++) {
		for(int q = 0; q < VDATA_SIZE; q++) {
			oW.data[i*VDATA_SIZE+q].range() = v[i].data[q].range();
		}
	}	
}

template <int lalala> void
consume(
    const v_arr *inW,            // Read-Only Weights
	hls::stream<v_arr128 > &oW_s
	)
{
	v_arr v[8];
	v_arr128 oW;
	#pragma HLS array_partition variable=v dim=0
	#pragma HLS array_partition variable=oW dim=0

	for(int iter=0; iter < Tsize*Nmat*Tsize/VDATA_SIZE/Mempaths; iter++) {
		v[iter%8] = inW[iter];
		if((iter%8) == 7) {
			makeone<lalala>(v, oW);
			oW_s.write(oW);
		}
	}
}

template <int lalala> void
writeone(v_arr v[64], hls::stream<v_arr1024 > &o_s)
{
	v_arr1024 oW;
	#pragma HLS array_partition variable=oW dim=0
	#pragma HLS pipeline II=1
	for(int i = 0; i < 64; i++) {
		for(int q = 0; q < VDATA_SIZE; q++) {
			oW.data[i*VDATA_SIZE+q].range() = v[i].data[q].range();
		}
	}	
	o_s.write(oW);
}

void
hbmV(
    const v_arr *inV,            // Read-Only Weights
	hls::stream<v_arr1024 > &oV_s0
	)
{
	v_arr v[Veclen][64];
	#pragma HLS array_partition variable=v dim=2
	#pragma HLS array_partition variable=v dim=3

	for(int i=0; i < 64; i++) {
		#pragma HLS pipeline II=1
		for(int q = 0; q < Veclen; q++) {
			v[q][i] = inV[i*Veclen+q];
		}
	}

	for(int iter=0; iter < Tsize*Nmat; iter++) {
		for(int q = 0; q < Veclen; q++) {
			#pragma HLS pipeline II=1
			writeone<0>(v[q], oV_s0);
		}
	}
}

void
math(
	hls::stream<v_arr1024 > &inV_s0,
	hls::stream<v_arr1024 > &inW_s0,
	hls::stream<ap_axiu<Itsize,0,0,0> > &o_s0
)
{
	Dt V[Tsize], W[Tsize];
	#pragma HLS array_partition variable=W dim=1
	#pragma HLS array_partition variable=V dim=1
	#pragma HLS pipeline enable_flush
	fetchW(inW_s0, W);
	fetchV(inV_s0, V);
	onemath(V, W, o_s0);
}

void
batchmath(
	hls::stream<v_arr1024 > &inV_s0,
	hls::stream<v_arr1024 > &inW_s0,
	hls::stream<ap_axiu<Itsize,0,0,0> > &o_s0
	)
{
	for(int i=0; i < Tsize*Veclen*Nmat; i++) {
		math(inV_s0, inW_s0, o_s0);
	}
}

void
replicateW(
	hls::stream<v_arr128 > &i_s0,
	hls::stream<v_arr128 > &i_s1,
	hls::stream<v_arr128 > &i_s2,
	hls::stream<v_arr128 > &i_s3,
	hls::stream<v_arr128 > &i_s4,
	hls::stream<v_arr128 > &i_s5,
	hls::stream<v_arr128 > &i_s6,
	hls::stream<v_arr128 > &i_s7,
	hls::stream<v_arr1024 > &o_s0
	)
{
	v_arr1024 v;
	v_arr128 iv[8];

	#pragma HLS array_partition variable=iv dim=0
	#pragma HLS array_partition variable=v dim=0

	for(int iter=0; iter < Tsize*Nmat; iter++) {
		#pragma HLS pipeline II=1
		i_s0.read(iv[0]);
		i_s1.read(iv[1]);
		i_s2.read(iv[2]);
		i_s3.read(iv[3]);
		i_s4.read(iv[4]);
		i_s5.read(iv[5]);
		i_s6.read(iv[6]);
		i_s7.read(iv[7]);

		for(int i = 0; i < 8; i++) {
			for(int j = 0; j < Tsize/8; j++) {
				v.data[i*Tsize/8+j] = iv[i].data[j];
			}
		}

		for(int r=0; r < Veclen; r++) {
			#pragma HLS pipeline II=1
			o_s0.write(v);
		}
	}
}

extern "C" {
void
feeder(
    const v_arr *inV,            // Read-Only Weights
    const v_arr *inW0,            // Read-Only Weights
    const v_arr *inW1,            // Read-Only Weights
    const v_arr *inW2,            // Read-Only Weights
    const v_arr *inW3,            // Read-Only Weights
    const v_arr *inW4,            // Read-Only Weights
    const v_arr *inW5,            // Read-Only Weights
    const v_arr *inW6,            // Read-Only Weights
    const v_arr *inW7,            // Read-Only Weights
    hls::stream<ap_axiu<Itsize,0,0,0> > &o_s0
)
{
	#pragma HLS INTERFACE m_axi port = inV offset = slave bundle = gmem16 max_read_burst_length=16 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = inW0 offset = slave bundle = gmem0 max_read_burst_length=16 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = inW1 offset = slave bundle = gmem1 max_read_burst_length=16 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = inW2 offset = slave bundle = gmem2 max_read_burst_length=16 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = inW3 offset = slave bundle = gmem3 max_read_burst_length=16 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = inW4 offset = slave bundle = gmem4 max_read_burst_length=16 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = inW5 offset = slave bundle = gmem5 max_read_burst_length=16 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = inW6 offset = slave bundle = gmem6 max_read_burst_length=16 max_write_burst_length=128
	#pragma HLS INTERFACE m_axi port = inW7 offset = slave bundle = gmem7 max_read_burst_length=16 max_write_burst_length=128
	#pragma HLS INTERFACE s_axilite bundle = control port = inV
	#pragma HLS INTERFACE s_axilite bundle = control port = inW0
	#pragma HLS INTERFACE s_axilite bundle = control port = inW1
	#pragma HLS INTERFACE s_axilite bundle = control port = inW2
	#pragma HLS INTERFACE s_axilite bundle = control port = inW3
	#pragma HLS INTERFACE s_axilite bundle = control port = inW4
	#pragma HLS INTERFACE s_axilite bundle = control port = inW5
	#pragma HLS INTERFACE s_axilite bundle = control port = inW6
	#pragma HLS INTERFACE s_axilite bundle = control port = inW7
	#pragma HLS INTERFACE s_axilite bundle = control port = return
	#pragma HLS INTERFACE axis port=o_s0

	#pragma HLS dataflow

	static hls::stream<v_arr1024 > oV_s0;
	static hls::stream<v_arr1024 > oW_sall;
	static hls::stream<v_arr128 > oW_s0;
	static hls::stream<v_arr128 > oW_s1;
	static hls::stream<v_arr128 > oW_s2;
	static hls::stream<v_arr128 > oW_s3;
	static hls::stream<v_arr128 > oW_s4;
	static hls::stream<v_arr128 > oW_s5;
	static hls::stream<v_arr128 > oW_s6;
	static hls::stream<v_arr128 > oW_s7;

	#pragma HLS STREAM variable=oV_s0 depth=2	
	#pragma HLS STREAM variable=oW_sall depth=2	
	#pragma HLS STREAM variable=oW_s0 depth=2	
	#pragma HLS STREAM variable=oW_s1 depth=2	
	#pragma HLS STREAM variable=oW_s2 depth=2	
	#pragma HLS STREAM variable=oW_s3 depth=2	
	#pragma HLS STREAM variable=oW_s4 depth=2	
	#pragma HLS STREAM variable=oW_s5 depth=2	
	#pragma HLS STREAM variable=oW_s6 depth=2	
	#pragma HLS STREAM variable=oW_s7 depth=2	

	consume<0>(inW0, oW_s0);
	consume<1>(inW1, oW_s1);
	consume<2>(inW2, oW_s2);
	consume<3>(inW3, oW_s3);
	consume<4>(inW4, oW_s4);
	consume<5>(inW5, oW_s5);
	consume<6>(inW6, oW_s6);
	consume<7>(inW7, oW_s7);
	hbmV(inV, oV_s0);
	replicateW(oW_s0,oW_s1,oW_s2,oW_s3,oW_s4,oW_s5,oW_s6,oW_s7, oW_sall);
	batchmath(oV_s0, oW_sall, o_s0);
}

} // extern C