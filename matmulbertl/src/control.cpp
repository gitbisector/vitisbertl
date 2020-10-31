#include "ap_axi_sdata.h"
#include "ap_int.h"
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
controlsubv(
	    const v_dt *in_tensor,           // Read-only tensor (1024, 14)
		hls::stream<v_arr> &ov00_s
		)
{
	v_dt W;
	for(int i=0; i < (Tsize*Veclen)/VDATA_SIZE; i++) {
		W = in_tensor[i];
		ov00_s.write(convWo(W));
	}
}

extern "C" {
void
control(
	    const v_dt *in_tensor,           // Read-only tensor (1024, 14)
		hls::stream<v_arr> &ov00_s
	    )
{
	#pragma HLS INTERFACE m_axi port = in_tensor offset = slave bundle = gmem16 
	#pragma HLS INTERFACE s_axilite bundle = cfg port = in_tensor
	#pragma HLS INTERFACE s_axilite bundle = cfg port = return
	#pragma HLS INTERFACE axis register_mode=both register port=ov00_s
	controlsubv(in_tensor, ov00_s);
}
}
