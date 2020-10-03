#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "qop.h"

extern "C" {
v_dt
convW(v_arr W) {
	#pragma HLS inline
	v_dt Wo;
	for(int i=0; i < VDATA_SIZE; i++)
		Wo.data[i] = W.data[i];
	return Wo;
}

void
wb(
    v_dt *o_tensor,          
    int shift,
	hls::stream<It> &i_s
    )
{
	#pragma HLS INTERFACE m_axi port=o_tensor offset=slave bundle=gmem16
	#pragma HLS INTERFACE axis register_mode=both register port=i_s
	#pragma HLS INTERFACE s_axilite port=o_tensor
	#pragma HLS INTERFACE s_axilite port=shift
	#pragma HLS INTERFACE s_axilite port = return

	It e;
	v_arr V;
	for(int i = 0; i < (Tsize*3*Veclen)/VDATA_SIZE; i++) {
		for(int k=0; k < VDATA_SIZE; k++) {
			i_s.read(e);
			V.data[k] = e >> shift;
		}
		o_tensor[i] = convW(V);
	}
}
}
