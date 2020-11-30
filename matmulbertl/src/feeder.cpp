#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "qop.h"
#include <iostream>

typedef struct v_in128 { Dt data[Tsize/8]; } v_arr128;

enum {
	Bsize = Tsize/8,
	BsizeP2 = Tp2-3, 
};

void
finalsum(
	hls::stream<It > &i_s0,
	hls::stream<It > &i_s1,
	hls::stream<It > &i_s2,
	hls::stream<It > &i_s3,
	hls::stream<It > &i_s4,
	hls::stream<It > &i_s5,
	hls::stream<It > &i_s6,
	hls::stream<It > &i_s7,
	hls::stream<It > &o_s
	)
{
	It s[8], o;
	#pragma HLS array_partition variable=s dim=0
	for(int iter=0; iter < Tsize*Veclen*Nmat; iter++) {
		#pragma HLS pipeline II=1
		i_s0.read(s[0]);
		i_s1.read(s[1]);
		i_s2.read(s[2]);
		i_s3.read(s[3]);
		i_s4.read(s[4]);
		i_s5.read(s[5]);
		i_s6.read(s[6]);
		i_s7.read(s[7]);
		o = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];
		o_s.write(o);
	}
}

void
onemath(It inV[Bsize], It W[Bsize], hls::stream<It > &o_s)
{
	It imm[BsizeP2+1][Bsize];
	#pragma HLS pipeline II=1
	#pragma HLS array_partition variable=imm dim=0
	l1: for(int b = 0; b < Bsize; b++) {
		imm[0][b] = (Dt)(inV[b] * W[b]);
	}
	for(int s = 0; s < BsizeP2; s++) {
		for(int k = 0; k < (Bsize>>(s+1)); k++) {
			imm[s+1][k] = imm[s][k] + imm[s][k+(Bsize>>(s+1))];
		}
	}
	o_s.write(imm[BsizeP2][0]);
}

void
makeone(v_arr v[8],	v_arr128 &oW)
{
	#pragma HLS pipeline II=1
	for(int i = 0; i < 8; i++) {
		for(int q = 0; q < VDATA_SIZE; q++) {
			oW.data[i*VDATA_SIZE+q].range() = v[i].data[q].range();
		}
	}	
}

void
replicate (
	hls::stream<v_arr128 > &i_s,
	hls::stream<v_arr128 > &o_s
	)
{
	v_arr128 v;
	for(int iter=0; iter < Tsize*Nmat; iter++) {
		#pragma HLS pipeline II=14
		i_s.read(v);
		for(int i=0; i < Veclen; i++) {
			o_s.write(v);
		}
	}
}

void
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
			makeone(v, oW);
			oW_s.write(oW);
		}
	}
}

void
hbmV(
    const v_arr *inV,            // Read-Only Weights
	hls::stream<v_arr128 > &oV_s0,
	hls::stream<v_arr128 > &oV_s1,
	hls::stream<v_arr128 > &oV_s2,
	hls::stream<v_arr128 > &oV_s3,
	hls::stream<v_arr128 > &oV_s4,
	hls::stream<v_arr128 > &oV_s5,
	hls::stream<v_arr128 > &oV_s6,
	hls::stream<v_arr128 > &oV_s7
	)
{
	v_arr128 v[Veclen][8];
	#pragma HLS array_partition variable=v dim=2
	#pragma HLS array_partition variable=v dim=3

	for(int q = 0; q < Veclen; q++) {
		for(int i=0; i < 64; i++) {
			#pragma HLS pipeline II=1
			v_arr t;
			t = inV[i+q*8*8];
			for(int z=0; z < VDATA_SIZE; z++)
				v[q][i>>3].data[(i%8)*16+z] = t.data[z];
		}
	}

	for(int iter=0; iter < Tsize*Nmat; iter++) {
		for(int q = 0; q < Veclen; q++) {
			#pragma HLS pipeline II=1
			oV_s0.write(v[q][0]);
			oV_s1.write(v[q][1]);
			oV_s2.write(v[q][2]);
			oV_s3.write(v[q][3]);
			oV_s4.write(v[q][4]);
			oV_s5.write(v[q][5]);
			oV_s6.write(v[q][6]);
			oV_s7.write(v[q][7]);
		}
	}
}

void
batchmath(
	hls::stream<v_arr128 > &inW_s0,
	hls::stream<v_arr128 > &inV_s0,
	hls::stream<It > &o_s0
	)
{
	v_arr128 V,W;
	It bV[Bsize], bW[Bsize];
	#pragma HLS array_partition variable=bV dim=0
	#pragma HLS array_partition variable=bW dim=0
	#pragma HLS array_partition variable=W dim=1
	#pragma HLS array_partition variable=V dim=1
	for(int i=0; i < Veclen*Tsize*Nmat; i++) {
		#pragma HLS pipeline
		inV_s0.read(V);
		inW_s0.read(W);
		for(int i = 0; i < Bsize; i++) {
			bV[i] = V.data[i];
			bW[i] = W.data[i];
		}
		onemath(bV, bW, o_s0);
	}
}

void
wb(
    int shift,
	hls::stream<It > &i_s0,
    wb_arr *o_tensor
    )
{
	It e;
	wb_arr V;
	#pragma HLS array_partition variable=V dim=0
	l_a: for(int i=0; i < (Nmat*Tsize*Veclen)/VDATA_SIZE; i++) {
		#pragma HLS pipeline II=16
		for(int q=0; q < VDATA_SIZE; q++) {
			i_s0.read(e);
			V.data[q] = e >> shift;
		}
		o_tensor[i] = V;
	}
}

extern "C" {
void
feeder(
    const v_arr *inV,             // Read-Only Weights
    const v_arr *inW0,            // Read-Only Weights
    const v_arr *inW1,            // Read-Only Weights
    const v_arr *inW2,            // Read-Only Weights
    const v_arr *inW3,            // Read-Only Weights
    const v_arr *inW4,            // Read-Only Weights
    const v_arr *inW5,            // Read-Only Weights
    const v_arr *inW6,            // Read-Only Weights
    const v_arr *inW7,            // Read-Only Weights
    wb_arr *o_tensor,
    int shift
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
	#pragma HLS INTERFACE m_axi port = o_tensor offset=slave bundle = gmem17 max_read_burst_length=16 max_write_burst_length=128
	#pragma HLS INTERFACE s_axilite bundle = control port = inV
	#pragma HLS INTERFACE s_axilite bundle = control port = inW0
	#pragma HLS INTERFACE s_axilite bundle = control port = inW1
	#pragma HLS INTERFACE s_axilite bundle = control port = inW2
	#pragma HLS INTERFACE s_axilite bundle = control port = inW3
	#pragma HLS INTERFACE s_axilite bundle = control port = inW4
	#pragma HLS INTERFACE s_axilite bundle = control port = inW5
	#pragma HLS INTERFACE s_axilite bundle = control port = inW6
	#pragma HLS INTERFACE s_axilite bundle = control port = inW7
	#pragma HLS INTERFACE s_axilite bundle = control port = o_tensor
	#pragma HLS INTERFACE s_axilite bundle = control port = shift
	#pragma HLS INTERFACE s_axilite bundle = control port = return

	#pragma HLS dataflow

	static hls::stream<v_arr128 > oV_s0;
	static hls::stream<v_arr128 > oV_s1;
	static hls::stream<v_arr128 > oV_s2;
	static hls::stream<v_arr128 > oV_s3;
	static hls::stream<v_arr128 > oV_s4;
	static hls::stream<v_arr128 > oV_s5;
	static hls::stream<v_arr128 > oV_s6;
	static hls::stream<v_arr128 > oV_s7;
	static hls::stream<v_arr128 > oW_s0;
	static hls::stream<v_arr128 > oW_s1;
	static hls::stream<v_arr128 > oW_s2;
	static hls::stream<v_arr128 > oW_s3;
	static hls::stream<v_arr128 > oW_s4;
	static hls::stream<v_arr128 > oW_s5;
	static hls::stream<v_arr128 > oW_s6;
	static hls::stream<v_arr128 > oW_s7;
	static hls::stream<v_arr128 > oWa_s0;
	static hls::stream<v_arr128 > oWa_s1;
	static hls::stream<v_arr128 > oWa_s2;
	static hls::stream<v_arr128 > oWa_s3;
	static hls::stream<v_arr128 > oWa_s4;
	static hls::stream<v_arr128 > oWa_s5;
	static hls::stream<v_arr128 > oWa_s6;
	static hls::stream<v_arr128 > oWa_s7;
	static hls::stream<It > o_i0;
	static hls::stream<It > o_i1;
	static hls::stream<It > o_i2;
	static hls::stream<It > o_i3;
	static hls::stream<It > o_i4;
	static hls::stream<It > o_i5;
	static hls::stream<It > o_i6;
	static hls::stream<It > o_i7;
	static hls::stream<It > o_wb;

	#pragma HLS STREAM variable=oV_s0 depth=3	
	#pragma HLS STREAM variable=oV_s1 depth=3	
	#pragma HLS STREAM variable=oV_s2 depth=3	
	#pragma HLS STREAM variable=oV_s3 depth=3	
	#pragma HLS STREAM variable=oV_s4 depth=3	
	#pragma HLS STREAM variable=oV_s5 depth=3	
	#pragma HLS STREAM variable=oV_s6 depth=3	
	#pragma HLS STREAM variable=oV_s7 depth=3	
	#pragma HLS STREAM variable=oW_s0 depth=2	
	#pragma HLS STREAM variable=oW_s1 depth=2	
	#pragma HLS STREAM variable=oW_s2 depth=2	
	#pragma HLS STREAM variable=oW_s3 depth=2	
	#pragma HLS STREAM variable=oW_s4 depth=2	
	#pragma HLS STREAM variable=oW_s5 depth=2	
	#pragma HLS STREAM variable=oW_s6 depth=2	
	#pragma HLS STREAM variable=oW_s7 depth=2	
	#pragma HLS STREAM variable=oWa_s0 depth=2	
	#pragma HLS STREAM variable=oWa_s1 depth=2	
	#pragma HLS STREAM variable=oWa_s2 depth=2	
	#pragma HLS STREAM variable=oWa_s3 depth=2	
	#pragma HLS STREAM variable=oWa_s4 depth=2	
	#pragma HLS STREAM variable=oWa_s5 depth=2	
	#pragma HLS STREAM variable=oWa_s6 depth=2	
	#pragma HLS STREAM variable=oWa_s7 depth=2	
	#pragma HLS STREAM variable=o_i0 depth=2	
	#pragma HLS STREAM variable=o_i1 depth=2	
	#pragma HLS STREAM variable=o_i2 depth=2	
	#pragma HLS STREAM variable=o_i3 depth=2	
	#pragma HLS STREAM variable=o_i4 depth=2	
	#pragma HLS STREAM variable=o_i5 depth=2	
	#pragma HLS STREAM variable=o_i6 depth=2	
	#pragma HLS STREAM variable=o_i7 depth=2	
	#pragma HLS STREAM variable=o_wb depth=2	

	hbmV(inV, oV_s0, oV_s1, oV_s2, oV_s3, oV_s4, oV_s5, oV_s6, oV_s7);
	consume(inW0, oW_s0);
	consume(inW1, oW_s1);
	consume(inW2, oW_s2);
	consume(inW3, oW_s3);
	consume(inW4, oW_s4);
	consume(inW5, oW_s5);
	consume(inW6, oW_s6);
	consume(inW7, oW_s7);
	replicate(oW_s0, oWa_s0);
	replicate(oW_s1, oWa_s1);
	replicate(oW_s2, oWa_s2);
	replicate(oW_s3, oWa_s3);
	replicate(oW_s4, oWa_s4);
	replicate(oW_s5, oWa_s5);
	replicate(oW_s6, oWa_s6);
	replicate(oW_s7, oWa_s7);
	batchmath(oWa_s0, oV_s0, o_i0);
	batchmath(oWa_s1, oV_s1, o_i1);
	batchmath(oWa_s2, oV_s2, o_i2);
	batchmath(oWa_s3, oV_s3, o_i3);
	batchmath(oWa_s4, oV_s4, o_i4);
	batchmath(oWa_s5, oV_s5, o_i5);
	batchmath(oWa_s6, oV_s6, o_i6);
	batchmath(oWa_s7, oV_s7, o_i7);
	finalsum(o_i0, o_i1, o_i2, o_i3, o_i4, o_i5, o_i6, o_i7, o_wb);
	wb(shift, o_wb, o_tensor);
}

} // extern C