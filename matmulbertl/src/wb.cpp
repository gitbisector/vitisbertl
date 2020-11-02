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
	hls::stream<ap_axiu<Itsize,0,0,0> > &i_s
    )
{
	It e;
	wb_arr V;
	It psums[Veclen][VDATA_SIZE];
	ap_axiu<Itsize,0,0,0> r;

	#pragma HLS INTERFACE m_axi port=o_tensor offset=slave bundle=gmem0
	#pragma HLS INTERFACE axis register_mode=both register port=i_s
	#pragma HLS INTERFACE s_axilite bundle = control port = o_tensor
	#pragma HLS INTERFACE s_axilite bundle = control port = shift
	#pragma HLS INTERFACE s_axilite bundle = control port = return
	#pragma HLS array_partition variable=psums dim=2

	for(int i=0; i < Nmat*Tsize/VDATA_SIZE; i++) {
		for(int k=0; k < VDATA_SIZE; k++) {
			for(int piter=0; piter < 8/cores; piter++) {
				for(int v=0; v < Veclen; v++) {
					i_s.read(r);
					e = (It)r.data;
					psums[v][k] = ((piter==0)?(It)0:psums[k]) + e;

					if((k == VDATA_SIZE-1) && (piter==8/cores-1)) {
						for(int k=0; k < VDATA_SIZE; k++) {
							V.data[k] = psums[v][k] >> shift;
						}
						o_tensor[i*Veclen+v] = V;
					}
				}
			}
		}
	}
}
}
