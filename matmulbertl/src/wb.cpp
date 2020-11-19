#include "ap_axi_sdata.h"
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "qop.h"

extern "C" {
void
wb(
    wb_arr *o_tensor,          
    int shift,
	hls::stream<ap_axiu<Itsize,0,0,0> > &i_s0,
	hls::stream<ap_axiu<Itsize,0,0,0> > &i_s1
    )
{
	It e0,e1;
	wb_arr V;
	It psums[Veclen][VDATA_SIZE];
	ap_axiu<Itsize,0,0,0> r0,r1;

	#pragma HLS INTERFACE m_axi port=o_tensor offset=slave bundle=gmem0
	#pragma HLS INTERFACE axis register_mode=both register port=i_s0
	#pragma HLS INTERFACE axis register_mode=both register port=i_s1
	#pragma HLS INTERFACE s_axilite bundle = control port = o_tensor
	#pragma HLS INTERFACE s_axilite bundle = control port = shift
	#pragma HLS INTERFACE s_axilite bundle = control port = return
	#pragma HLS array_partition variable=psums dim=2

	for(int i=0; i < Nmat*Tsize/VDATA_SIZE; i++) {
		for(int k=0; k < VDATA_SIZE; k++) {
			for(int piter=0; piter < 8/cores/2; piter++) {
				for(int v=0; v < Veclen; v++) {
					i_s0.read(r0);
					e0(e0.width-1,0) = r0.data;
					i_s1.read(r1);
					e1(e1.width-1,0) = r1.data;
					psums[v][k] = ((piter==0)?(It)0:psums[v][k]) + e0 + e1;

					if((k == VDATA_SIZE-1) && (piter==8/cores/2-1)) {
						for(int z=0; z < VDATA_SIZE; z++) {
							#pragma HLS pipeline II=1
							V.data[z] = psums[v][z] >> shift;
						}
						o_tensor[i*Veclen+v] = V;
					}
				}
			}
		}
	}
}
}
