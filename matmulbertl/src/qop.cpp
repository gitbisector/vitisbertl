#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "qop.h"



v_arr
convWo(v_dt W) {
	#pragma HLS inline
	v_arr Wo;
	for(int i=0; i < VDATA_SIZE; i++)
		Wo.data[i] = W.data[i];
	return Wo;
}

void
rmem(int iter, const v_dt *inW, v_arr W[Mopers/VDATA_SIZE])
{
	v_dt iW;
#pragma HLS pipeline II=14
	for(int z = 0; z < Mopers/VDATA_SIZE; z++) {
		iW = inW[iter*(Mopers/VDATA_SIZE)+z];
		W[z] = convWo(iW);
	}
}

void
math(int iter, v_arr inV[Mopers/VDATA_SIZE][Veclen][(Tsize/Mopers)/cores], v_arr W[Mopers/VDATA_SIZE], hls::stream<ap_axiu<Itsize,0,0,0> > &o_s)
{
	It res;
	It imm[MopersP2+1][Mopers];
	#pragma HLS pipeline II=14
	#pragma HLS array_partition variable imm dim=0
	#pragma HLS allocation instances=fmul limit=128 operation
	#pragma HLS allocation instances=fadd limit=128 operation
	#pragma HLS allocation instances=mul  limit=128 operation
	#pragma HLS allocation instances=hmul limit=128 operation
	#pragma HLS allocation instances=hadd limit=128 operation

	for(int v = 0; v < Veclen; v++) {
		l4: for(int b = 0; b < Mopers/VDATA_SIZE; b++) {
			v_arr x;
			x = inV[b][v][iter % ((Tsize/Mopers)/cores)];
			l3:for(int k = 0 ; k < VDATA_SIZE; k++) {
				imm[0][b*VDATA_SIZE+k] = (Dt)(W[b].data[k] * x.data[k]);
			}
		}
		for(int s = 0; s < MopersP2; s++) {
			for(int k = 0; k < (Mopers>>(s+1)); k++) {
				imm[s+1][k] = imm[s][k] + imm[s][k+(Mopers>>(s+1))];
			}
		}
		ap_axiu<Itsize,0,0,0> o_s_v;
		o_s_v.data = imm[MopersP2][0];
		o_s.write(o_s_v);
	}
}

extern "C" {
void
qop(
    const v_dt *inW,            // Read-Only Weights
	hls::stream<v_arr> &inV_s,
	hls::stream<ap_axiu<Itsize,0,0,0> > &o_s
)
{
	v_arr W[Mopers/VDATA_SIZE];
	v_arr inV[Mopers/VDATA_SIZE][Veclen][(Tsize/Mopers)/cores];

#pragma HLS INTERFACE m_axi port = inW offset = slave bundle = gmem0 max_read_burst_length=128 max_write_burst_length=128
#pragma HLS INTERFACE axis port=inV_s
#pragma HLS INTERFACE axis register_mode=both register port=o_s

#pragma HLS resource core=RAM_S2P_BRAM variable=inV
#pragma HLS array_partition variable=inW dim=1
#pragma HLS array_partition variable=inV dim=1
#pragma HLS array_partition variable=inV dim=4
#pragma HLS array_partition variable=W dim=1

#pragma HLS INTERFACE s_axilite bundle = control port = inW
#pragma HLS INTERFACE s_axilite bundle = control port = return

	vecloop: for(int v = 0; v < Veclen; v++) {
		for(int iter = 0; iter < (Tsize/Mopers)/cores; iter++) {
			for(int b = 0; b < Mopers/VDATA_SIZE; b++) {
				#pragma HLS pipeline II=1
				inV_s.read(inV[b][v][iter]);
			}
		}
	}

#pragma HLS dataflow
	weightloop: for(int iter = 0; iter < ((Tsize*Tsize*Nmat)/Mopers)/cores; iter++) {
		rmem(iter, inW, W);
		math(iter, inV, W, o_s);
	}
}
}
