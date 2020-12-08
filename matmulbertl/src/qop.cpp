#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "qop.h"
#include <iostream>

static v_arr
convV(ap_axiu<256,0,0,0> W) {
	#pragma HLS inline
	v_arr Wo;
	//std::cout << std::hex << W.data << std::endl;
	for(int i=0; i < VDATA_SIZE; i++) {
		Wo.data[i](Wo.data[0].width-1,0) = W.data(Wo.data[0].width*(i+1)-1, Wo.data[0].width*i);
	}
	return Wo;
}

template <int lalala> void
rmem(int iter, const v_arr *inW, v_arr W[Mopers/VDATA_SIZE])
{
	v_arr iW;
	#pragma HLS pipeline II=8
	for(int z = 0; z < Mopers/VDATA_SIZE; z++) {
		iW = inW[iter*(Mopers/VDATA_SIZE)+z];
		W[z] = iW;
	}
}

template <int lalala> void
onemath(int iter, int v, v_arr inV[Mopers/VDATA_SIZE][Veclen][(Tsize/Mopers)/cores][Qop_paths], v_arr W[Mopers/VDATA_SIZE][Qop_paths], hls::stream<ap_axiu<Itsize,0,0,0> > &o_s)
{
	It res;
	It imm[MopersP2+1+Qop_pathsP2][Mopers*Qop_paths];
	#pragma HLS pipeline II=1
	#pragma HLS array_partition variable=imm dim=0
	#pragma HLS allocation instances=fmul limit=1024 operation
	#pragma HLS allocation instances=fadd limit=1024 operation
	#pragma HLS allocation instances=mul  limit=1024 operation
	#pragma HLS allocation instances=hmul limit=1024 operation
	#pragma HLS allocation instances=hadd limit=1024 operation
	l4: for(int b = 0; b < Mopers/VDATA_SIZE; b++) {
		for(int q = 0; q < Qop_paths; q++) {
			v_arr x, w;
			x = inV[b][v][iter % ((Tsize/Mopers)/cores)][q];
			w = W[b][q];
			l3:for(int k = 0; k < VDATA_SIZE; k++) {
				imm[0][(b*Qop_paths+q)*VDATA_SIZE+k] = (Dt)(w.data[k] * x.data[k]);
			}
		}
	}
	for(int s = 0; s < MopersP2+Qop_pathsP2; s++) {
		for(int k = 0; k < ((Mopers*Qop_paths)>>(s+1)); k++) {
			imm[s+1][k] = imm[s][k] + imm[s][k+((Mopers*Qop_paths)>>(s+1))];
		}
	}
	ap_axiu<Itsize,0,0,0> o_s_v;
	o_s_v.data(imm[0][0].width-1, 0) = imm[MopersP2+Qop_pathsP2][0](imm[0][0].width-1, 0);
	o_s.write(o_s_v);
}

template <int lalala> void
math(int iter, v_arr inV[Mopers/VDATA_SIZE][Veclen][(Tsize/Mopers)/cores][Qop_paths], v_arr W[Mopers/VDATA_SIZE][Qop_paths], hls::stream<ap_axiu<Itsize,0,0,0> > &o_s)
{
	#pragma HLS pipeline II=14
	math_l: for(int v = 0; v < Veclen; v++) {
		onemath<0>(iter, v, inV, W, o_s);
	}
}

void
rmemarr(
	int iter,
    hls::stream<ap_axiu<512,512,1,1> > &i_A,
    hls::stream<ap_axiu<512,512,1,1> > &i_B,
	v_arr W0[Mopers/VDATA_SIZE][Qop_paths],
	int &r
)
{
	#pragma HLS pipeline II=8
	ap_axiu<512,512,1,1> A,B;
	for(int z = 0; z < Mopers/VDATA_SIZE; z++) {
		i_A.read(A);
		i_B.read(B);
		for(int i = 0; i < VDATA_SIZE; i++) {
			#pragma HLS pipeline II=1
			W0[0][z].data[i](15,0) = A.data(    (i+1)*16-1,     i*16);
			W0[1][z].data[i](15,0) = A.data(256+(i+1)*16-1, 256+i*16);
			W0[2][z].data[i](15,0) = A.user(    (i+1)*16-1,     i*16);
			W0[3][z].data[i](15,0) = A.user(256+(i+1)*16-1, 256+i*16);
			W0[4][z].data[i](15,0) = B.data(    (i+1)*16-1,     i*16);
			W0[5][z].data[i](15,0) = B.data(256+(i+1)*16-1, 256+i*16);
			W0[6][z].data[i](15,0) = B.user(    (i+1)*16-1,     i*16);
			W0[7][z].data[i](15,0) = B.user(256+(i+1)*16-1, 256+i*16);
		}
	}
	r = iter;
}

extern "C" {
void
qop(
    hls::stream<ap_axiu<512,512,1,1> > &i_A,
    hls::stream<ap_axiu<512,512,1,1> > &i_B,
	hls::stream<ap_axiu<256,0,0,0> > &inV_s,
	hls::stream<ap_axiu<Itsize,0,0,0> > &o_s0
)
{
	ap_axiu<256,0,0,0> inV;

	#pragma HLS INTERFACE axis port=inV_s
	#pragma HLS INTERFACE axis port=i_A
	#pragma HLS INTERFACE axis port=i_B
	#pragma HLS INTERFACE axis register_mode=both register port=o_s0

	v_arr inV0[Mopers/VDATA_SIZE][Veclen][(Tsize/Mopers)/cores][Qop_paths];

	#pragma HLS array_partition variable=inV0 dim=1
	#pragma HLS array_partition variable=inV0 dim=3
	#pragma HLS array_partition variable=inV0 dim=4
	#pragma HLS array_partition variable=inV0 dim=5
	#pragma HLS INTERFACE s_axilite bundle = control port = return

	vecloop: for(int v = 0; v < Veclen; v++) {
		for(int q= 0; q < Qop_paths; q++) {
			for(int iter = 0; iter < (((Tsize/Mopers)/cores))/Qop_paths; iter++) {
				for(int b = 0; b < Mopers/VDATA_SIZE; b++) {
					#pragma HLS pipeline II=1
					inV_s.read(inV);
					inV0[b][v][iter][q] = convV(inV);
				}
			}
		}
	}

	weightloop: for(int iter = 0; iter < (((Tsize*Tsize*Nmat)/Mopers)/cores)/Qop_paths; iter++) {
		int a;
		v_arr W0[Mopers/VDATA_SIZE][Qop_paths];
		#pragma HLS dataflow
		#pragma HLS array_partition variable=W0 dim=1
		#pragma HLS array_partition variable=W0 dim=2
		#pragma HLS array_partition variable=W0 dim=3
		rmemarr(iter, i_A, i_B, W0, a);
		math<0>(a, inV0, W0, o_s0);
	}
}
}
