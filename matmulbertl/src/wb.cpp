#include "ap_axi_sdata.h"
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "qop.h"
#include <iostream>

extern "C" {
void
wb(
    wb_arr *o_tensor,          
    int shift,
	hls::stream<ap_axiu<Itsize,0,0,0> > &i_s0
    )
{
	It e0,e1;
	wb_arr V;
	static It psums[Veclen][VDATA_SIZE];
	ap_axiu<Itsize,0,0,0> r;

	#pragma HLS INTERFACE m_axi port=o_tensor offset=slave bundle=gmem0
	#pragma HLS INTERFACE axis register_mode=both register port=i_s0
	#pragma HLS INTERFACE axis register_mode=both register port=i_s1
	#pragma HLS INTERFACE s_axilite bundle = control port = o_tensor
	#pragma HLS INTERFACE s_axilite bundle = control port = shift
	#pragma HLS INTERFACE s_axilite bundle = control port = return
	#pragma HLS array_partition variable=psums dim=2

	l_a: for(int i=0; i < Nmat*Tsize/VDATA_SIZE; i++) {
		l_b: for(int k=0; k < VDATA_SIZE; k++) {
			l_c: for(int piter=0; piter < (8/cores)/Qop_paths; piter++) {
				l_d: for(int v=0; v < Veclen; v++) {
					i_s0.read(r);
					e(e.width-1,0) = r.data(e.width-1,0);

					if(e != (It)0)
						std::cout << (int)e(31,0) << std::endl;

					if((k == VDATA_SIZE-1) && (piter==(8/cores)/Qop_paths-1)) {
						l_e: for(int m=0; m < VDATA_SIZE; m++) {
							V.data[m] = (psums[v][m] + ((m==k)?e:(It)0)) >> shift;
							psums[v][m] = 0;
						}
						o_tensor[i*Veclen+v] = V;
					}
					else
						psums[v][k] += e;
				}
			}
		}
	}
}
}

