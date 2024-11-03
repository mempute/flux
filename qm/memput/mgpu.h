#pragma once

#include <cuda_runtime.h>
#include "memput.h"
using namespace memput::mp;

#define WIDTH_BLOCK					(BLOCK_SIZE > idx_width ? idx_width : BLOCK_SIZE)
#define WIDTH_BLOCK2(block_size)	(block_size > idx_width ? idx_width : block_size)
#define WIDTH_BLOCK3(k_size)		(k_size > BLOCK_SIZE ? BLOCK_SIZE : k_size)


extern intt gsplit_f(void *pcxt, floatt *m_split, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width,
	intt nsplit, intt nstep, intt axis, bool bw);
extern intt gsplit_f(void *pcxt, intt *m_split, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width,
	intt nsplit, intt nstep, intt axis, bool bw);
extern intt gconcat_f(void *pcxt, floatt *m_rcat, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width,
	intt ncat, intt nstep, intt axis, bool bw);
extern intt gconcat_f(void *pcxt, intt *m_rcat, intt pdim, intt sdim, intt rsize, intt idx_origin, intt idx_width,
	intt ncat, intt nstep, intt axis, bool bw);
extern intt gdot_f(void *pcxt_dev, intt oksz, intt share_unit, floatt *pdot_mdev, floatt *sdot_mdev, floatt *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, floatt rplus);
extern intt gdot_f(void *pcxt_dev, intt oksz, intt share_unit, intt *pdot_mdev, intt *sdot_mdev, intt *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, intt rplus);
extern intt gdot_f2(void *pcxt_dev, floatt *pdot_mdev, floatt *sdot_mdev, floatt *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, floatt rplus);
extern intt gdot_f2(void *pcxt_dev, intt *pdot_mdev, intt *sdot_mdev, intt *rdot_mdev, intt rdot_size,
	intt idx_origin, intt idx_width, intt rplus);
extern intt garith_f(void *pcxt, floatt *m_pari, floatt *m_sari, floatt *m_rari, intt p_size, intt r_size,
	intt idx_origin, intt idx_width, floatt sval, sytet aop, floatt rplus, sytet tp_arith, sytet bw);
extern intt garith_f(void *pcxt, intt *m_pari, intt *m_sari, intt *m_rari, intt p_size, intt r_size,
	intt idx_origin, intt idx_width, intt sval, sytet aop, intt rplus, sytet tp_arith, sytet bw);
extern intt gtrans_f(void *pcxt, floatt *m_strs, floatt *m_rtrs, intt r_size, intt idx_origin, intt idx_width, bool bw);
extern intt gtrans_f(void *pcxt, intt *m_strs, intt *m_rtrs, intt r_size, intt idx_origin, intt idx_width, bool bw);
extern intt gsoftx_f(void *pcxt, floatt *m_ssfx, floatt *m_rsfx, floatt *m_sum, floatt *m_max, floatt *m_buf, intt r_size, intt f_size, sytet db, intt idx_origin, intt idx_width);
extern intt gsoftx_f(void *pcxt, intt *m_ssfx, intt *m_rsfx, intt *m_sum, intt *m_max, intt *m_buf, intt r_size, intt f_size, sytet db, intt idx_origin, intt idx_width);
extern intt gsoftx_cross_e_f(void *pcxt, floatt *m_ssfx, floatt *m_rsfx, floatt *m_tsfx, intt r_size, intt f_size, intt idx_origin, intt idx_width);
extern intt gsoftx_cross_e_f(void *pcxt, intt *m_ssfx, intt *m_rsfx, intt *m_tsfx, intt r_size, intt f_size, intt idx_origin, intt idx_width);
extern intt gmse_f(void *pcxt, floatt *m_smet, floatt *m_tmet, floatt *m_rmet, intt r_size, intt idx_origin, intt idx_width, bool mean);
extern intt gmse_f(void *pcxt, intt *m_smet, intt *m_tmet, intt *m_rmet, intt r_size, intt idx_origin, intt idx_width, bool mean);
extern intt gsum_f(void *pcxt, floatt *m_smet, floatt *m_rmet, intt r_size, intt idx_origin, intt idx_width, floatt *cmul, bool mean);
extern intt gsum_f(void *pcxt, intt *m_smet, intt *m_rmet, intt r_size, intt idx_origin, intt idx_width, intt *cmul, bool mean);
extern intt gbsum_f(void *pcxt, floatt *mpre, floatt *mret, intt r_size, intt idx_origin, intt idx_width, intt sum_sz, bool mean, bool bw, sytet rplus);
extern intt gbsum_f(void *pcxt, intt *mpre, intt *mret, intt r_size, intt idx_origin, intt idx_width, intt sum_sz, bool mean, bool bw, sytet rplus);
extern intt goptadm_f(void *pcxt, floatt *mm, floatt *mv, floatt *mg, floatt *mr, intt r_size, intt idx_origin,
	intt idx_width, floatt beta1, floatt beta2, floatt lr, floatt ep, intt dec, sytet db);
extern intt goptadm_f(void *pcxt, intt *mm, intt *mv, intt *mg, intt *mr, intt r_size, intt idx_origin,
	intt idx_width, intt beta1, intt beta2, intt lr, intt ep, intt dec, sytet db);
extern intt goptsgd_f(void *pcxt, floatt *mg, floatt *mr, intt r_size, intt idx_origin, intt idx_width, floatt lr, intt dec);
extern intt goptsgd_f(void *pcxt, intt *mg, intt *mr, intt r_size, intt idx_origin, intt idx_width, intt lr, intt dec);
extern intt gactf_f(void *pcxt, floatt *mpre, floatt *msuf, floatt *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db);
extern intt gactf_f(void *pcxt, intt *mpre, intt *msuf, intt *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db);
extern intt gactf2_f(void *pcxt, floatt *mpre, floatt *msuf, floatt *mret, floatt *m1, floatt *m2, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db);
extern intt gactf2_f(void *pcxt, intt *mpre, intt *msuf, intt *mret, intt *m1, intt *m2, intt r_size, intt idx_origin, intt idx_width, intt aop2, sytet rplus, sytet db);
extern intt gtwo_f(void *pcxt, floatt *mpre, floatt *msuf, floatt *mret, floatt *bpre, floatt *bsuf, intt r_size, intt idx_origin, intt idx_width, intt aop2);
extern intt gtwo_f(void *pcxt, intt *mpre, intt *msuf, intt *mret, intt *bpre, intt *bsuf, intt r_size, intt idx_origin, intt idx_width, intt aop2);
extern intt gembedding_f(floatt *mpre, floatt *msuf, floatt *mret, intt r_size, intt idx_origin, intt idx_width, intt sz_embed, intt stp, intt etable_sz, intt bw);
extern intt gembedding_f(intt *mpre, intt *msuf, intt *mret, intt r_size, intt idx_origin, intt idx_width, intt sz_embed, intt stp, intt etable_sz, intt bw);
extern intt gonehot_f(void *pcxt, floatt *mpre, floatt *mret, intt r_size, intt idx_origin, intt idx_width);
extern intt gonehot_f(void *pcxt, intt *mpre, intt *mret, intt r_size, intt idx_origin, intt idx_width);
extern intt gslice_f(void *pcxt, floatt *mpre, floatt *mret, intt r_size, intt idx_origin, intt idx_width, bool bw, sytet rplus);
extern intt gslice_f(void *pcxt, intt *mpre, intt *mret, intt r_size, intt idx_origin, intt idx_width, bool bw, sytet rplus);
extern intt gargmax_f(void *pcxt, floatt *mpre, floatt *mret, intt r_size, intt idx_origin, intt idx_width);
extern intt gargmax_f(void *pcxt, intt *mpre, intt *mret, intt r_size, intt idx_origin, intt idx_width);
extern intt gvmax_f(void *pcxt, floatt *mpre, floatt *msuf, floatt *mret, intt r_size, intt idx_origin, intt idx_width, intt bw, sytet rplus);
extern intt gvmax_f(void *pcxt, intt *mpre, intt *msuf, intt *mret, intt r_size, intt idx_origin, intt idx_width, intt bw, sytet rplus);
extern intt gequal_f(void *pcxt, floatt *mpre, floatt *msuf, floatt *mret, intt r_size, intt idx_origin, intt idx_width);
extern intt gequal_f(void *pcxt, intt *mpre, intt *msuf, intt *mret, intt r_size, intt idx_origin, intt idx_width);
extern intt gtype1_f(void *pcxt, floatt *mpre, floatt *msuf, floatt *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2);
extern intt gtype1_f(void *pcxt, intt *mpre, intt *msuf, intt *mret, intt r_size, intt idx_origin, intt idx_width, intt aop2);
extern intt grandom_f(void *pcxt, floatt *mpre, intt r_size, intt idx_origin, intt idx_width, intt aop2, intt seed);
extern intt grandom_f(void *pcxt, intt *mpre, intt r_size, intt idx_origin, intt idx_width, intt aop2, intt seed);
extern intt glayer_norm_f(void *pcxt, floatt *mi, floatt *mr, floatt *md, floatt *mz, floatt *mv, floatt *mean,
	floatt *g_mz, floatt *var, floatt *mdmean, floatt *ga, floatt *be, floatt *g_gm, floatt *g_be,
	intt r_size, intt idx_origin, intt idx_width, intt dsz, bool bw, bool db);
extern intt glayer_norm_f(void *pcxt, intt *mi, intt *mr, intt *md, intt *mz, intt *mv, intt *mean,
	intt *g_mz, intt *var, intt *mdmean, intt *ga, intt *be, intt *g_gm, intt *g_be,
	intt r_size, intt idx_origin, intt idx_width, intt dsz, bool bw, bool db);
extern intt gmatmul_f(void *pcxt, floatt mpre[], floatt msuf[], floatt mret[], intt r_size, intt M, intt K, intt N, intt T, bool rplus,
	intt idx_origin, intt idx_width);
extern intt gmatmul_f(void *pcxt, intt mpre[], intt msuf[], intt mret[], intt r_size, intt M, intt K, intt N, intt T, bool rplus,
	intt idx_origin, intt idx_width);
