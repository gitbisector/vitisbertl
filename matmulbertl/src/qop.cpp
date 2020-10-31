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

extern "C" {
void
qop(
    const v_dt *inW,            // Read-Only Weights
	hls::stream<v_arr> &inV_s,
	hls::stream<ap_axiu<Itsize,0,0,0> > &o_s
)
{
	v_dt iW;
	v_arr W[Mopers/VDATA_SIZE];
	v_arr inV[Mopers/VDATA_SIZE][Veclen][(Tsize/Mopers)/cores];
	It res;

#pragma HLS INTERFACE m_axi port = inW offset = slave bundle = gmem0 max_read_burst_length=128 max_write_burst_length=128
#pragma HLS INTERFACE axis port=inV_s
#pragma HLS INTERFACE axis register_mode=both register port=o_s
#pragma HLS allocation instances=fmul limit=128 operation
#pragma HLS allocation instances=fadd limit=128 operation
#pragma HLS allocation instances=mul  limit=128 operation
#pragma HLS allocation instances=hmul limit=128 operation
#pragma HLS allocation instances=hadd limit=128 operation
#pragma HLS resource core=RAM_S2P_BRAM variable=inV
#pragma HLS array_partition variable=inW dim=1
#pragma HLS array_partition variable=inV dim=1
#pragma HLS array_partition variable=inV dim=4
#pragma HLS array_partition variable=W dim=1

#pragma HLS INTERFACE s_axilite bundle = control port = inW
#pragma HLS INTERFACE s_axilite bundle = control port = return

	for(int v = 0; v < Veclen; v++) {
		for(int iter = 0; iter < (Tsize/Mopers)/cores; iter++) {
			for(int b = 0; b < Mopers/VDATA_SIZE; b++) {
				#pragma HLS pipeline II=1
				inV_s.read(inV[b][v][iter]);
			}
		}
	}

	weightloop: for(int iter = 0; iter < ((Tsize*Tsize*Nmat)/Mopers)/cores; iter++) {
		#pragma HLS pipeline II=14
		for(int z = 0; z < Mopers/VDATA_SIZE; z++) {
			iW = inW[iter*(Mopers/VDATA_SIZE)+z];
			W[z] = convWo(iW);
		}
		for(int v = 0; v < Veclen; v++) {
			l4: for(int b = 0; b < Mopers/VDATA_SIZE; b++) {
				v_arr x;
				x = inV[b][v][iter % ((Tsize/Mopers)/cores)];
				l3:for(int k = 0 ; k < VDATA_SIZE; k++) {
					res = (((k==0) && (b==0))?(It)0:res) + (Dt)(W[b].data[k] * x.data[k]);
				}
			}
			ap_axiu<Itsize,0,0,0> o_s_v;
			o_s_v.data = res;
			o_s.write(o_s_v);
		}
	}
}
}
