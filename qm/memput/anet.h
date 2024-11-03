#pragma once
#include "trace.h"

#define STRIDE_DOT 0
#define TENSOR_DOT 1
#define COUPLE_DOT 2
#define TRIPPLE_DOT 3
#define ORTHO_DOT	4
#define REDUCE_DOT	5

#define C_XT 0
#define C_ZT 1
#define C_ZZ 2

typedef struct _FluxChain {
	Flux *ptrFlux;
	_FluxChain *ptrLeft, *ptrRight;
} FluxChain;

class Anet : public Typer {
public:
	Tracer *tcrAnt;
	floatt rExContigAnt;
	intt latszAnt, kernszAnt, strideAnt, widthAnt, yDiscrete, szmhead;
	sytet lat_tAnt, actfAnt, dot_tAnt, cmodAnt, zpadAnt;
	bool elasticAnt, intervAnt, finalReduced;
	FluxChain *lfhstate;
	Flux *t_hid, *entrygate, *encodeOut, *batchLoss;
	bool prtbuild;
	Flux *iCode, *ovCode, *ppCode, *teternal, *reternal;
	Flux *stepcode, *bycode;
	//agzo.zpad를 1로 하면 입력 만큼만 조합전개되므로 입력이 커널 사이즈에 정합되지 않을 경우 마지막 스트라이드
	//구간의 조합 갯수가 이전 스트라이드 조합개수보다 적어 일정한 갯수로 행렬 처리할 수가 없다. 3,4번은 입력 시퀀스에서 모자르는
	//부분을 제로 패딩하여 이로서 조합을 생성하므로 패딩값이 조합 패턴으로 생성되므로 입력이 왜곡될수있다. 2번은 실 입력값까지만
	//조합을 생성하고 마지막 스트라이드에서 모자르는 조합 패턴분(량)을 단순 제로패딩하므로 나머지 조합은 모두 0이 되어 학습과정에서
	//무시될수있으므로(가중치애 기울기 값에 영향이 없으므로) 입력값이 왜곡되지 않는다. 
	//그러나 2번은 expand batch가 아닌 combination 함수로만 실행될수있으므로 중간층에 사용되면 그 이하로 역전파되지 않는다.
	//따라서 2번은 convolving이 아니고 정합되지 않는 게이스에 옵션을 주어 사용한다. 정합되면 제로패딩 필요없으므로 상관없다.
	void optAnet(intt lookup_sz = 8, intt step = 8, sytet zpad = 4, floatt rexc = 0.7, sytet dot_t = STRIDE_DOT, bool el = 0)
	{
		Trace *trc = TRACER(tcrAnt);

		kernszAnt = (trc->kernelSize < 0 ? lookup_sz : trc->kernelSize); //커널 윈도우 사이즈
		strideAnt = (trc->strideJump < 0 ? widthAnt : trc->strideJump);
		rExContigAnt = (trc->contigusRate < 0 ? rexc : trc->contigusRate);
		zpadAnt = (trc->zeroPadding < 0 ? zpad : trc->zeroPadding);
		dot_tAnt = (trc->dotType < 0 ? dot_t : trc->dotType);
		elasticAnt = (trc->elasticHidden < 0 ? el : trc->elasticHidden);

		if(widthAnt == strideAnt) intervAnt = 0;
		else intervAnt = 1;
		cmodAnt = -1;
	}
	Anet(Tracer *tcr, intt latent_sz = -1, sytet af = ACTF_TANH)
	{
		tcrAnt = tcr;
		widthAnt = TRACER(tcrAnt)->bandwidth;
		latszAnt = latent_sz;  //잠재 히든 유닛 노드 갯수, 이하 3개 변수가 학습에 영향을 많이 미침.
		actfAnt = af;
		yDiscrete = 0;
		lfhstate = nullx;
		entrygate = nullx;
		prtbuild = TRACER(tcrAnt)->printBuilding;
		finalReduced = 0;
		szmhead = 0;
		iCode = ovCode = ppCode = teternal = reternal = nullx;
	}
	Flux *expand_batch(Flux *ins, Flux *mask, bool inter_leave, intt kernsz) //[batch, seq, in feat]
	{
		intt n_devide_seq = ins->fshape[1] / kernsz, mul, width;
		Flux *cat_ins;

		if(kernsz < widthAnt) {//커널 사이즈가 완전조합 시퀀스 길이 보다 적으면 커널사이즈를 폭으로 설정
			width = kernsz;
			mul = 1;
		} else if(kernsz % widthAnt) throwFault(-1, "not aligned kernel\n");
		else {
			width = widthAnt;
			mul = kernsz / widthAnt;
		}
		if(inter_leave) {
			intt half = kernsz / 2;
			Flux *pad = flux(tcrAnt, { ins->fshape[0], half, ins->fshape[2] }, ins->qType, constant);
			pad->adjust(ins);
			auto shift_ins = concat({ ins->slice({ {}, {half, -1} }), pad }, 1);
			//shift_ins->printo(1, 2);
			//printf("--------------1----------------\n");
			ins = ins->reshape({ -1, n_devide_seq, width, ins->fshape[2] });//[batch, ins_devide_seq, width, feat]
			shift_ins = shift_ins->reshape({ -1, n_devide_seq, width, shift_ins->fshape[2] });//[batch, shift_devide_seq, width, feat]
			//ins->printo(1, 2);
			//ins->shape();
			//shift_ins->printo(1, 2);
			//shift_ins->shape();
			//printf("--------------3----------------\n");
			cat_ins = stack({ ins, shift_ins }, 2);//seq차원(1차원)아래 차원하나를 더 만들어(2차원) 양측 시퀀스(1차원)를 한개씩 묶어 새로생성한 (2)차원에 넣음
			//cat_ins->printo(1, 2);//[batch, 2, (ins+shift)party, width, feat]
			//cat_ins->shape();
			cat_ins = cat_ins->reshape({ -1, cat_ins->fshape[1] * cat_ins->fshape[2], cat_ins->fshape[3], cat_ins->fshape[4] });
			//cat_ins->printo(1, 2);//[batch, party, width, feat]
			//cat_ins->shape();
		} else {
			//printf('--------------3----------------');
			cat_ins = ins->reshape({ -1, n_devide_seq, width, ins->fshape[2] });//[batch, party, width, feat]
			//cat_ins->printo(1, 2);
			//cat_ins->shape();
		}
		//cat_ins = cat_ins->bypass("11\n");
		//printf("--------------4----------------\n");
		cat_ins = cat_ins->expand_dims(2);//[batch, party, 1, width, feat]
		//cat_ins->printo(1, 2);
		//cat_ins->shape();
		//printf("--------------5----------------\n");
		cat_ins = *cat_ins * *mask;//[batch, party, derive, width, feat]=[batch, party, 1, width, feat] * [derive, width, feat]
		//cat_ins->printo(1, 2);
		//cat_ins->shape();
		//printf("--------------6----------------\n");
		//cat_ins = cat_ins->bypass("22\n");
		if(mul > 1) {//커널사이즈가 완전조합 길이보다 크면 완전조합 시퀀스 길이를 커널사이즈에 맟주고
			/* //width horiz concat example//그 배수만큼 완전조합 길이 단위로 분리되는 갯수를 줄인다.
			by = flux(tcr, { 2, 4, 3, 2, 2 }, tfloat, constant);
			by->arange(-1);
			by->printo();
			by = by->transpose({ 0, 1, 3, 2, 4 });
			by = by->reshape({ 2, 2, 4, 3, 2 });
			by = by->transpose({ 0, 1, 3, 2, 4 });
			by->printo();*/
			cat_ins = cat_ins->transpose({ 0, 1, 3, 2, 4 });//[batch, party, width, derive, feat]
			cat_ins = cat_ins->reshape({ -1, cat_ins->fshape[1] / mul, ins->fshape[2] * mul, cat_ins->fshape[3], ins->fshape[4] });
			//[batch, reduced_party, kernel, derive, feat]
			cat_ins = cat_ins->transpose({ 0, 1, 3, 2, 4 });//[batch, reduced_party, derive, kernel, feat]
		}
		cat_ins = cat_ins->reshape({ -1, cat_ins->fshape[1] * cat_ins->fshape[2], cat_ins->fshape[3], cat_ins->fshape[4] });
		//cat_ins->printo(1, 2);//[batch, bindbatch(party*derive), kernel, feat]
		//cat_ins->shape();
		return cat_ins;
	}
	Flux *expand_batch2(Flux *ins, Flux *mask, intt stride, intt kernsz, sytet tzpad, intt &outsz) //[batch, seq, in feat]
	{
		intt nrest, nstride, nseq, mul, width;
		//행렬 연산 방식으로는 사이즈가 맞지 않으면 expand할수없으므로 패딩을 한다. 패딩 옵션은 outer(tzpad == 4)가 아니면
		//모두 inner padding으로 처리한다.
		if(kernsz < widthAnt) {//커널 사이즈가 완전조합 시퀀스 길이 보다 적으면 커널사이즈를 폭으로 설정
			width = kernsz;
			mul = 1;
		} else if(kernsz % widthAnt) throwFault(-1, "not aligned kernel\n");
		else {
			width = widthAnt;
			mul = kernsz / widthAnt;
		}
		if(tzpad == 4) {//outer zero padding. 시퀀스 32, 커널8, 스트라이드 6 인경우 마지막 스트라이드는 30이고 
			nseq = ins->fshape[1];//시퀀스 사이즈 32에서 모자르는 6개를 (8 - (32 - 30))을 제로패딩한다.
			nstride = nseq / stride;
			if(nseq % stride == 0) nstride--;
			nrest = nstride * stride;
			nrest = nrest + width - nseq;
		} else {//inner zero padding. 시퀀스 32, 커널8, 스트라이드 6 인경우 마지막 스트라이드는 24이고
			nseq = ins->fshape[1] - width;//24 + 8(커널) == 32 시퀀스 사이즈에 정합되므로 제로패딩없이 끝.
			nstride = nseq / stride;
			if(nseq % stride) {
				nstride++;
				nrest = nstride * stride + width - ins->fshape[1];
			} else nrest = 0;
		}
		if(TRACER(tcrAnt)->derivefeat == 2 && outsz > 0) {//시퀀스가 8 ~ 32사이 이면 32로 확장하여 
			if(ins->fshape[1] > widthAnt && ins->fshape[1] < widthAnt * 4) {//압축결과 4가 리턴되게 한다.
				outsz = 4;
				nseq = widthAnt * outsz;
				nrest = nseq - ins->fshape[1];
				nstride = nseq / stride;
				if(nseq % stride == 0) nstride--;
			}
		}
		if(nrest) {//width의 끝으로부터 시퀀스의 끝까지 구간이 스트라이드 사이즈 단위에 정합되지 않으면 마지막에 모자르는 구간을 제로
			Flux *pad = flux(tcrAnt, { ins->fshape[0], nrest, ins->fshape[2] }, ins->qType, constant);//패딩하여 더한다.
			pad->adjust(ins);
			ins = concat({ ins, pad }, 1);//[batch, (seq + pad), in feat]
		}
		if(nstride) {
			vector<Flux *> fxl;
			for(intt i = 0; i <= nstride; i++) {
				fxl.push_back(ins->slice({ {}, {i * stride, i * stride + width} }));
			}
			ins = stack(&fxl, 1);//[batch, party, width, feat]
		} else ins = ins->expand_dims(1);//[batch, 1, width, feat]
		//ins->printo(1, 2);
		//ins->shape();
		ins = ins->expand_dims(2);//[batch, party, 1, width, feat]
		//ins->printo(1, 2);
		//ins->shape();
		//printf("--------------1----------------\n");
		ins = *ins * *mask;//[batch, party, derive, width, feat]=[batch, party, 1, width, feat] * [derive, width, feat]
		//ins->printo(1, 2);
		//ins->shape();
		//printf("--------------2----------------\n");
		//ins = ins->bypass("22\n");
		if(mul > 1) {//커널사이즈가 완전조합 길이(widthAnt)보다 크면 완전조합 시퀀스 길이를 커널사이즈에 맟주고
			/* //width horiz concat example//그 배수만큼 완전조합 길이 단위로 분리되는 갯수(party)를 줄인다.
			by = flux(tcr, { 2, 4, 3, 2, 2 }, tfloat, constant);
			by->arange(-1);
			by->printo();
			by = by->transpose({ 0, 1, 3, 2, 4 });
			by = by->reshape({ 2, 2, 4, 3, 2 });
			by = by->transpose({ 0, 1, 3, 2, 4 });
			by->printo();*/
			ins = ins->transpose({ 0, 1, 3, 2, 4 });//[batch, party, width, derive, feat]
			ins = ins->reshape({ -1, ins->fshape[1] / mul, ins->fshape[2] * mul, ins->fshape[3], ins->fshape[4] });
			//[batch, reduced_party, kernel, derive, feat]
			ins = ins->transpose({ 0, 1, 3, 2, 4 });//[batch, reduced_party, derive, kernel, feat]
		}
		ins = ins->reshape({ -1, ins->fshape[1] * ins->fshape[2], ins->fshape[3], ins->fshape[4] });
		//ins->printo(1, 2);//[batch, bindbatch(party*derive), kernel, feat]
		//ins->shape();
		return ins;
	}
	//[batch, seq, sizing] = [batch, seq, hidden] * [seq, hidden, sizing] + [seq, sizing]
	Flux *coupledot(Flux *inp, Flux *wsizing, Flux *wb)
	{
		auto uns_g = inp->unstack(1);//seq_sz *[batch, hidden](one - seq)
		auto uns_w = wsizing->unstack(0);//seq_sz *[hidden, sizing](one - seq)
		auto uns_b = wb->unstack(0);//seq_sz *[sizing](one - seq)
		vector<Flux *> l_uns;
		for(intt t = 0; t < uns_g->size(); t++) {
			auto _uns_g = uns_g->at(t);
			auto _uns_w = uns_w->at(t);
			auto _uns_b = uns_b->at(t);
			auto g_out = *_uns_g->dot(_uns_w, { {1}, {0} }) + *_uns_b;//{[batch, sizing] = [batch, hidden] * [hidden, sizing] + [sizing]}(one - seq)
			l_uns.push_back(g_out);
		}
		return stack(&l_uns, 1);
	}
	//스트라이드 단위로 가중치 공유
	//[batch, party, seq, sizing] = [batch, party, seq, hidden] * [seq, hidden, sizing] + [seq, sizing]
	Flux *trippledot(Flux *inp, Flux *wsizing, Flux *wb)
	{
		auto uns_g = inp->unstack(2); //seq_sz *[batch, party, hidden](one - seq)
		auto uns_w = wsizing->unstack(0); //seq_sz *[hidden, sizing](one - seq)
		auto uns_b = wb->unstack(0); //seq_sz *[sizing](one - seq)
		vector<Flux *> l_uns;
		for(intt t = 0; t < uns_g->size(); t++) {
			auto _uns_g = uns_g->at(t);
			auto _uns_w = uns_w->at(t);
			auto _uns_b = uns_b->at(t);
			auto g_out = *_uns_g->dot(_uns_w, { {2}, {0} }) + *_uns_b;// {[batch, party, sizing] = [batch, party, hidden] * [hidden, sizing] + [sizing]}(one - seq)
			l_uns.push_back(g_out);
		}
		return stack(&l_uns, 2);
	}
	Flux *trippledot2(Flux *inp, Flux *wsizing) //편향 없음
	{
		auto uns_g = inp->unstack(2); //seq_sz *[batch, party, hidden](one - seq)
		auto uns_w = wsizing->unstack(0); //seq_sz *[hidden, sizing](one - seq)
		vector<Flux *> l_uns;
		for(intt t = 0; t < uns_g->size(); t++) {
			auto _uns_g = uns_g->at(t);
			auto _uns_w = uns_w->at(t);
			auto g_out = _uns_g->dot(_uns_w, { {2}, {0} });// {[batch, party, sizing] = [batch, party, hidden] * [hidden, sizing]
			l_uns.push_back(g_out);
		}
		return stack(&l_uns, 2);
	}
	//[batch, party, seq, sizing] = [batch, party, seq, hidden] * [party, hidden, sizing] + [party, sizing]
	Flux *stridedot(Flux *inp, Flux *wsizing, Flux *wb, const bytet *name = "stride_dot")
	{
		Trace *trc = TRACER(tcrAnt);
		if(trc->softmdot) trc->namescope(name);

		auto uns_g = inp->unstack(1); //party *[batch, seq, hidden](one - party)
		auto uns_w = wsizing->unstack(0); //party *[hidden, sizing](one - party)
		auto uns_b = wb->unstack(0); //party *[sizing](one - party)
		vector<Flux *> l_uns;
		for(intt t = 0; t < uns_g->size(); t++) {
			auto _uns_g = uns_g->at(t);
			auto _uns_w = uns_w->at(t);
			auto _uns_b = uns_b->at(t);
			if(trc->softmdot) {
				bytet s[1024];
				sprintf(s, "softm_x_%d", t);
				auto x = _uns_g->layer_dense(_uns_g->fshape[_uns_g->fdim - 1], actfAnt, Initializer::xavier, s);
				auto y = x->softmax();
				auto _uns_g = *x * *y;//[batch, seq, hidden](one - part)
			}
			auto g_out = *_uns_g->dot(_uns_w, { {2}, {0} }) + *_uns_b;// {[batch, seq, sizing] = [batch, seq, hidden] * [hidden, sizing] + [sizing]}(one - part)
			l_uns.push_back(g_out);
		}
		if(trc->softmdot) trc->endscope();
		return stack(&l_uns, 1);
	}
	//[batch, kernel, party, seq, sizing] = [batch, kernel, party, seq, hidden] * [party, hidden, sizing] + [party, sizing]
	Flux *dual_stridedot(Flux *inp, Flux *wsizing, Flux *wb, const bytet *name = "dual_sdot")
	{
		Trace *trc = TRACER(tcrAnt);
		if(trc->softmdot) trc->namescope(name);

		auto uns_g = inp->unstack(2); //party *[batch, kernel, seq, hidden](one - party)
		auto uns_w = wsizing->unstack(0); //party *[hidden, sizing](one - party)
		auto uns_b = wb->unstack(0); //party *[sizing](one - party)
		vector<Flux *> l_uns;
		for(intt t = 0; t < uns_g->size(); t++) {
			auto _uns_g = uns_g->at(t);
			auto _uns_w = uns_w->at(t);
			auto _uns_b = uns_b->at(t);
			if(trc->softmdot) {
				bytet s[1024];
				sprintf(s, "dsoftm_x_%d", t);
				auto x = _uns_g->layer_dense(_uns_g->fshape[_uns_g->fdim - 1], actfAnt, Initializer::xavier, s);
				auto y = x->softmax();
				auto _uns_g = *x * *y;//[batch, kernel, seq, hidden](one - part)
			}
			auto g_out = *_uns_g->dot(_uns_w, { {3}, {0} }) + *_uns_b;// {[batch, kernel, seq, sizing] = [batch, kernel, seq, hidden] * [hidden, sizing] + [sizing]}(one - part)
			l_uns.push_back(g_out);
		}
		if(trc->softmdot) trc->endscope();
		return stack(&l_uns, 2);
	}
	/*//[batch, kernel, party, seq, sizing] = [batch, kernel, party, seq, hidden] * [party, hidden, sizing] + [party, sizing]
	Flux *softmax_dot(Flux *inp, intt axis, const bytet *name = "softm_dot")
	{
		Trace *trc = TRACER(tcrAnt);
		trc->namescope(name);
		auto uns_g = inp->unstack(axis); //party *[batch, kernel, seq, hidden](one - party)
		auto uns_w = wsizing->unstack(0); //party *[hidden, sizing](one - party)
		auto uns_b = wb->unstack(0); //party *[sizing](one - party)
		vector<Flux *> l_uns;
		for(intt t = 0; t < uns_g->size(); t++) {
			auto _uns_g = uns_g->at(t);
			auto _uns_w = uns_w->at(t);
			auto _uns_b = uns_b->at(t);
			auto x = _uns_g->layer_dense(_uns_g->fshape[_uns_g->fdim -1], actfAnt, Initializer::xavier, "Q");
			auto y = x->softmax();
			auto g_out = *x * *y;//[batch, kernel, seq, hidden](one - part)
			l_uns.push_back(g_out);
		}
		trc->endscope();
		return stack(&l_uns, axis);
	}*/
	/*Flux *softmax_fillter(Flux *inp, intt out_sz, const bytet *name = "softm_fillter") //[batch, seq, feat]
	{
		Trace *trc = TRACER(tcrAnt);
		const bytet act_op = trc->actfAtt < 0 ? actfAnt : trc->actfAtt;
		trc->namescope(name);
		Flux *Q = inp->layer_dense(inp->fshape[1], act_op, Initializer::xavier, "Q");//[batch, seq, seq]
		Flux *V = inp->layer_dense(out_sz, act_op, Initializer::xavier, "V");//[batch, seq, out_sz]

		Flux *attention_weight = Q->softmax();//[batch, seq, seq]
		//[batch, seq, out_sz] = [batch, seq, seq] * [batch, seq, out_sz]
		auto r = attention_weight->matmul(V);
		trc->endscope();
		return r;
	}*/
	Flux *attention_layer(Flux *inp, intt out_sz, const bytet *name = "att_layer") //[batch, seq, feat]
	{	//[batch, seq, out_sz] = [batch, seq, feat] * [feat, out_sz]
		Trace *trc = TRACER(tcrAnt);
		const bytet act_op = trc->actfAtt < 0 ? actfAnt : trc->actfAtt;
		Flux *r;

		trc->namescope(name);
		if(trc->softfill) {
			Flux *Q = inp->layer_dense(inp->fshape[2], act_op, Initializer::xavier, "Q");//[batch, seq, feat]
			Flux *V = inp->layer_dense(inp->fshape[2], act_op, Initializer::xavier, "V");//[batch, seq, feat]
			auto S = flux(tcrAnt, { inp->fshape[2], out_sz }, lat_tAnt, constant);//[feat, out_sz]

			Flux *attention_weight = Q->softmax();//[batch, seq, feat]
			r = *attention_weight * *V;//[batch, seq, feat]
			S->fill(1.0);
			r = r->dot(S, { {2}, {0} });//[batch, seq, out_sz] = [batch, seq, feat] * [feat, out_sz]
		} else {
			Flux *Q = inp->layer_dense(out_sz, act_op, Initializer::xavier, "Q");
			Flux *K = inp->layer_dense(out_sz, act_op, Initializer::xavier, "K");
			Flux *V = inp->layer_dense(out_sz, act_op, Initializer::xavier, "V");

			Flux *cross_mul = Q->matmul(K, 2);//[batch, seq, seq] = [batch, seq, out_sz] * [batch, out_sz, seq]
			Flux *scaled = *cross_mul / std::sqrt(K->fshape[K->fdim - 1]);//[batch, seq, seq]
			Flux *attention_weight = scaled->softmax();//[batch, seq, seq]
			//[batch, seq, out_sz] = [batch, seq, seq] * [batch, seq, out_sz]
			r = attention_weight->matmul(V);
		}
		trc->endscope();
		return r;
	}
	Flux *attention_layer2(Flux *q_inp, Flux *k_inp, intt out_sz, const bytet *name = "att_layer2") //[batch, q_seq, feat]
	{	//[batch, q_seq, out_sz] = [batch, q_seq, feat] * [batch, k_seq, feat].T * [batch, k_seq, out_sz]
		Trace *trc = TRACER(tcrAnt);
		const bytet act_op = trc->actfAtt < 0 ? actfAnt : trc->actfAtt;
		trc->namescope(name);
		Flux *Q = q_inp->layer_dense(q_inp->fshape[2], act_op, Initializer::xavier, "Q");
		Flux *K = k_inp->layer_dense(k_inp->fshape[2], act_op, Initializer::xavier, "K");
		Flux *V = k_inp->layer_dense(out_sz, act_op, Initializer::xavier, "V");

		Flux *cross_mul = Q->matmul(K, 2);//[batch, q_seq, k_seq] = [batch, q_seq, feat] * [batch, k_seq, feat].T
		Flux *scaled = *cross_mul / std::sqrt(K->fshape[K->fdim - 1]);//[batch, q_seq, k_seq]
		Flux *attention_weight = scaled->softmax();//[batch, q_seq, k_seq]
		//[batch, q_seq, out_sz] = [batch, q_seq, k_seq] * [batch, k_seq, out_sz]
		auto r = attention_weight->matmul(V);
		trc->endscope();
		return r;
	}
	Flux *attention_dot(Flux *inp, intt out_sz, intt n_part) //[batch, party, seq, hidden]
	{
		Trace *trc = TRACER(tcrAnt);
		if(n_part == 1) return attention_layer(inp, out_sz);
		if(trc->positional > 0 && trc->rderivet == 0) {//aetpr.입력이 포지셔널되있고 party간 공유하여 어텐션 수행
			intt party = inp->fshape[1], seq = inp->fshape[2], hidden = inp->fshape[3];
			inp = inp->reshape({ -1, seq, hidden });//[batch*party, seq, hidden]
			inp = attention_layer(inp, out_sz);//[batch*party, seq, out_sz]
			return inp->reshape({ -1, party, seq, out_sz });//[batch, party, seq, out_sz]
		}
		auto uns_g = inp->unstack(1); //party *[batch, seq, hidden](one - party)
		bytet name[64];
		vector<Flux *> l_uns;
		for(intt t = 0; t < uns_g->size(); t++) {
			sprintf(name, "att_dot_%d\n", t);
			auto _uns_g = uns_g->at(t);
			auto g_out = attention_layer(_uns_g, out_sz, name);//[batch, seq, out_sz]
			l_uns.push_back(g_out);
		}
		return stack(&l_uns, 1);//[batch, party, seq, out_sz]
	}
	Flux *dualatten_dot(Flux *inp, intt out_sz, intt kernsz, intt n_part, intt ni_part) 
	{//inp - [batch, (kernel*party), seq, hidden]
		Flux *wcross = (ni_part ? flux(tcrAnt, { inp->fshape[3], out_sz }, lat_tAnt, constant, nullx, "wdua_cross") : nullx);
		Flux *g_out;
		vector<Flux *> l_uns;
		bytet name[64];

		auto uns_g = inp->unstack(1); //(kernel*party) *[batch, seq, hidden](one - party)
		if(ni_part) wcross->fill(1.0);//입력 파트에서는 커널의 마지막 스텝만 사용되므로 이것만 
		//어텐션 곱하고 입력파트에서 나머지 스텝은 단순 리덕션한다.(사용되지않으므로 역전파되지 않는다.)
		for(intt t = 0; t < uns_g->size(); t++) {
			sprintf(name, "dualatt_dot_%d\n", t);
			auto _uns_g = uns_g->at(t);//[batch, seq, hidden](one - party)
			if((t / n_part < ni_part) && ((t + 1) % kernsz)) g_out = _uns_g->dot(wcross, { {2}, {0} });
			else g_out = attention_layer(_uns_g, out_sz, name);//[batch, seq, out_sz]
			l_uns.push_back(g_out);
		}
		return stack(&l_uns, 1);//[batch, (kernel*party), seq, out_sz]
	}
	Flux *fillter_dot(Flux *inp, intt n_fillter)
	{//inp - [batch, party, (kernel), derive, latent]
		inp = inp->layer_dense(n_fillter, -1);
		inp = inp->vmax(inp->fdim - 2);
		return inp;//[batch, party, (kernel), n_fillter]
	}
	Flux *embedding(Flux *inp, intt vocab_size, intt embed_size, bool zero_pad = 1)//[batch, bindbatch, kernel]
	{
		Flux *embed_lookup = flux(tcrAnt, { vocab_size, embed_size }, lat_tAnt, trainable, Initializer::xavier, "embed_lookup");
		if(zero_pad) {
			auto a = flux(tcrAnt, { 1, embed_size }, lat_tAnt, constant);
			a->fill(0.0);
			embed_lookup = concat({ a, embed_lookup->slice({{1, -1}, {}}) }, 0);
		}
		auto out = embed_lookup->embedding_lookup(inp);//[batch, bindbatch, kernel, embed_size]
		return out;
	}
	Flux *expand_hook(Flux *inp, intt kernsz, intt latsz, intt featsz, intt n_part, intt n_derive)
	{
		Trace *trc = TRACER(tcrAnt);
		vector<Flux *> party_cat;
		//inp == [batch, party, derive, kernel, feat]
		if(trc->poskern == 2) {
			auto wkern = flux(tcrAnt, { n_part, kernsz, kernsz }, lat_tAnt, trainable, Initializer::xavier, "ehook_wkern");
			auto wkb = flux(tcrAnt, { n_part, 1, kernsz }, lat_tAnt, trainable, Initializer::xavier, "ehook_wkb");
			inp = inp->transpose({ 0, 1, 2, 4, 3 });//[batch, party, derive, feat, kernel]
			auto uns_inputs = inp->unstack(1);//[party](t_length) *[batch, derive, feat, kernel](one - seq),
			auto uns_kern = wkern->unstack(0);//[party](t_length) *[kernel, kernel](one - seq),
			auto uns_bk = wkb->unstack(0);//[party](t_length) *[1, kernel](one - seq),
			for(intt t = 0; t < uns_inputs->size(); t++) {
				auto input_t = uns_inputs->at(t);//[batch, derive, feat, kernel]
				auto _wkern = uns_kern->at(t);//[kernel kernel]
				auto _kb = uns_bk->at(t);//[1 kernel]
				auto x = *input_t->dot(_wkern, { {3}, {0} }) + *_kb;//[batch, derive, feat, kernel]=[batch, derive, feat, kernel]*[kernel kernel]
				party_cat.push_back(x);
			}
			inp = concat(&party_cat, 2);//[batch, derive, (party * feat), kernel]
			inp = inp->reshape({ -1, n_derive, n_part, featsz, kernsz });//[batch, derive, party, feat, kernel]
			inp = inp->transpose({ 0, 2, 1, 4, 3 });//[batch, party, derive, kernel, feat]
			inp = inp->actf(actfAnt);
		}
		/*//[batch, bindbatch(party*derive), kernel, featsz]
		Flux *exphook;
		if(e_hook == 1) {//[bindbatch(party*derive), kernel, featsz] broadcast
			exphook = flux(tcrAnt, inp->fdim - 1, &inp->fshape[1], lat_tAnt, trainable, Initializer::xavier, "wexp_hook");
			inp = *inp * *exphook;
		} else if(e_hook == 2) {//[bindbatch(party*derive), kernel] broadcast
			inp = inp->transpose({ 0, 3, 1, 2 });//[batch, featsz, bindbatch(party*derive), kernel]
			exphook = flux(tcrAnt, inp->fdim - 2, &inp->fshape[2], lat_tAnt, trainable, Initializer::xavier, "wexp_hook");
			inp = *inp * *exphook;
			inp = inp->transpose({ 0, 2, 3, 1 });//[batch, bindbatch(party*derive), kernel, featsz]
		} else if(e_hook == 3) {//bindbatch(party*derive) broadcast
			inp = inp->transpose({ 0, 2, 3, 1 });//[batch, kernel, featsz, bindbatch(party*derive)]
			exphook = flux(tcrAnt, 1, &inp->fshape[3], lat_tAnt, trainable, Initializer::xavier, "wexp_hook");
			inp = *inp * *exphook;
			inp = inp->transpose({ 0, 3, 1, 2 });//[batch, bindbatch(party*derive), kernel, featsz]
		} else {//derive broadcast
			intt party = inp->fshape[1] / n_derive;//[batch, party, derive, kernel, featsz]
			inp = inp->reshape({ -1, party, n_derive, inp->fshape[2], inp->fshape[3] });
			inp = inp->transpose({ 0, 1, 3, 4, 2 });//[batch, party, kernel, featsz, derive]
			exphook = flux(tcrAnt, 1, &inp->fshape[4], lat_tAnt, trainable, Initializer::xavier, "wexp_hook");
			inp = *inp * *exphook;
			inp = inp->transpose({ 0, 1, 4, 2, 3 });//[batch, party, derive, kernel, featsz]
			inp = inp->reshape({ -1, party * n_derive, inp->fshape[3], inp->fshape[4] });//[batch, bindbatch(party*derive), kernel, featsz]
		}
		inp = inp->actf(actfAnt);*/
		return inp;
	}
	Flux *kerneling(Flux *inp, bool dual, intt derivefeat, intt kernsz, intt latsz, intt featsz,
		intt n_bind, intt n_part, intt n_derive, intt iname)
	{
		Trace *trc = TRACER(tcrAnt);
		sytet actf_code = actfAnt, pos_kern = (trc->enckern == 100 || dual ? trc->poskern : trc->enckern);
		//완전조합은 커널스텝을 건너뛰는 것을 포함하므로 위치를 설정하기위해 가중치를 커널스텝별로 따로 한다.
		//스트라이드간이 위치는 wcross가 스트라이드별로 따로 있고 이를 스트라이드 곱을 하여 확보한다.
		//wxh는 위치 분별 기능이므로 derive를 나눠 커널링할때 각 derive분리간에 공유될수있도록 iname을 음수로한다.
		Flux *wxh, *whh, *wh, *bh, **hiddens, *hidden, *state;
		vector<Flux *> hid_cat;
		intt t = 0;//시간축(커널사이즈)을 따라 마지막 축(커널끝)까지 합해지므로 중간에 0입력들은(제로패딩포함) 무시되어 예측성능이 좋아진다.
		//inp == [batch, party, derive, kernel, featsz]
		if(pos_kern > 0) {
			if(pos_kern == 6) {//whh는 모든 커널 스텝별로 가중치를 두어 위치 분별하고 wxh는 커널 스텝별로 두어 
				if(prtbuild) printf("kernel expand 6\n");//party단위로 공유한다.
				Flux *_whh = nullx, *_wxh = nullx, *_bh = nullx;
				wxh = flux(tcrAnt, { kernsz, featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { n_part * kernsz, latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				bh = flux(tcrAnt, { kernsz, 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
				inp = inp->reshape({ -1, n_derive, n_part * kernsz, featsz });//[batch, derive, party * kernel, featsz]
				hiddens = (Flux **)trc->xalloc((n_part * kernsz) * sizeof(Flux *));
				//inp == [batch, derive, party * kernel, featsz]
				auto unstacked_inputs = inp->unstack(2);//[party * kernel](t_length) *[batch, derive, featsz](one - seq),
				auto unstacked_wxh = wxh->unstack(0);//[kernel](t_length) *[featsz latent](one - seq),
				auto unstacked_whh = whh->unstack(0);//[party * kernel](t_length) *[latent latent](one - seq),
				auto unstacked_bh = bh->unstack(0);//[kernel](t_length) *[1 latent](one - seq),
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, derive, featsz]
					_whh = unstacked_wxh->at(t);//[featsz latent]
					_wxh = unstacked_whh->at(t % kernsz);//[latent latent]
					_bh = unstacked_bh->at(t % kernsz);//[1 latent]
					if(t % kernsz) hidden = hiddens[t - 1];
					else {//hidden - [batch, derive, latent],각 party의 시작 t에서 초기값 히든을~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~설정하여 party간에 이전값이 곱해지지않게 한다.
					}
					auto x = input_t->dot(_wxh, { {2}, {0} });//[batch, derive, latent]=[batch, derive, featsz]*[featsz latent]
					auto h = hidden->dot(_whh, { {2}, {0} });//[batch, derive, latent] = [batch, derive, latent][latent, latent]
					auto a = *(*x + *h) + *_bh;//[batch, derive, latent]=[batch, derive, latent]+[batch, derive, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//각 커널의 마지막 스텝 히든 적재
				}
			} else if(pos_kern == 5) {//wxh와 whh를 party별로 두어 party단위로 공유한다.
				if(prtbuild) printf("kernel expand 5\n");//TRUE RIGHT. 3번보다 0.001정도 못하나 공유된다는 측면에서 더 낫지 않을까
				Flux *_wh = nullx, *_bh = nullx;
				wxh = flux(tcrAnt, { n_part, featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { n_part, latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				wh = concat({ wxh, whh }, 1);//[party, (featsz+latent) latent]
				bh = flux(tcrAnt, { n_part, 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
				inp = inp->reshape({ -1, n_derive, n_part * kernsz, featsz });//[batch, derive, party * kernel, featsz]
				hiddens = (Flux **)trc->xalloc((n_part * kernsz) * sizeof(Flux *));
				//inp == [batch, derive, party * kernel, featsz]
				auto unstacked_inputs = inp->unstack(2);//[party * kernel](t_length) *[batch, derive, featsz](one - seq),
				auto unstacked_wh = wh->unstack(0);//[party](t_length) *[(featsz+latent) latent](one - seq),
				auto unstacked_bh = bh->unstack(0);//[party](t_length) *[1 latent](one - seq),
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, derive, featsz]
					if(t % kernsz) hidden = hiddens[t - 1];
					else {//hidden - [batch, derive, latent],각 party의 시작 t에서 초기값 히든을~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~설정하여 party간에 이전값이 곱해지지않게 한다.
						_wh = unstacked_wh->at(t / kernsz);//[featsz+latent latent]
						_bh = unstacked_bh->at(t / kernsz);//[1 latent]
					}
					auto concat_x = concat({ input_t, hidden }, 2);//[batch, derive, (featsz + latent)] = [batch, derive, featsz](1) + [batch, derive, latent]
					auto a = *concat_x->dot(_wh, { {2}, {0} }) + *_bh;//[batch, derive, latent] = [batch, derive, (featsz + latent)][(featsz + latent), latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//각 커널의 마지막 스텝 히든 적재
				}
			} else if(pos_kern == 4) {//62.모든 커널 스텝별로 가중치를 두어 위치 분별
				if(prtbuild) printf("kernel expand 4\n");
				wxh = flux(tcrAnt, { n_part * kernsz, featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { n_part * kernsz, latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				wh = concat({ wxh, whh }, 1);//[party * kernel, (featsz+latent) latent]
				bh = flux(tcrAnt, { n_part * kernsz, 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
				inp = inp->reshape({ -1, n_derive, n_part * kernsz, featsz });//[batch, derive, party * kernel, featsz]
				hiddens = (Flux **)trc->xalloc((n_part * kernsz) * sizeof(Flux *));
				//inp == [batch, derive, party * kernel, featsz], wh = [party * kernel, (featsz+latent) latent]
				auto unstacked_inputs = inp->unstack(2);//[party * kernel](t_length) *[batch, derive, featsz](one - seq),
				auto unstacked_wh = wh->unstack(0);//[party * kernel](t_length) *[(featsz+latent) latent](one - seq),
				auto unstacked_bh = bh->unstack(0);
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, derive, featsz]
					auto _wh = unstacked_wh->at(t);//[(featsz + latent) latent]
					auto _bh = unstacked_bh->at(t);
					if(t % kernsz) hidden = hiddens[t - 1];
					else {//각 party의 시작 t에서 초기값 히든을~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~설정하여 party간에 이전값이 곱해지지않게 한다.
					}//hidden - [batch, derive, latent]
					//_wh = concat({ _wxh, _whh }, 0);//[(featsz + latent) latent] = [featsz latent] + [latent, latent]
					auto concat_x = concat({ input_t, hidden }, 2);//[batch, derive, (featsz + latent)] = [batch, derive, featsz](1) + [batch, derive, latent]
					auto a = *concat_x->dot(_wh, { {2}, {0} }) + *_bh;//[batch, derive, latent] = [batch, derive, (featsz + latent)][(featsz + latent), latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//각 커널의 마지막 스텝 히든 적재
				}
			} else if(pos_kern == 3) {//TRUE RIGHT.wxh는 모든 커널 스텝별로 가중치를 두어 위치 분별하고 whh는 party별로 두어 
				if(prtbuild) printf("kernel expand 3\n");//party단위로 공유한다.
				Flux *_whh = nullx, *_bh = nullx;
				wxh = flux(tcrAnt, { n_part * kernsz, featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { n_part, latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				bh = flux(tcrAnt, { n_part, 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
				inp = inp->reshape({ -1, n_derive, n_part * kernsz, featsz });//[batch, derive, party * kernel, featsz]
				hiddens = (Flux **)trc->xalloc((n_part * kernsz) * sizeof(Flux *));
				//inp == [batch, derive, party * kernel, featsz]
				auto unstacked_inputs = inp->unstack(2);//[party * kernel](t_length) *[batch, derive, featsz](one - seq),
				auto unstacked_wxh = wxh->unstack(0);//[party * kernel](t_length) *[featsz latent](one - seq),
				auto unstacked_whh = whh->unstack(0);//[party](t_length) *[latent latent](one - seq),
				auto unstacked_bh = bh->unstack(0);//[party](t_length) *[1 latent](one - seq),
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, derive, featsz]
					auto _wxh = unstacked_wxh->at(t);//[featsz latent]
					if(t % kernsz) hidden = hiddens[t - 1];
					else {//hidden - [batch, derive, latent],각 party의 시작 t에서 초기값 히든을~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~설정하여 party간에 이전값이 곱해지지않게 한다.
						_whh = unstacked_whh->at(t / kernsz);//[latent latent]
						_bh = unstacked_bh->at(t / kernsz);//[1 latent]
					}
					auto x = input_t->dot(_wxh, { {2}, {0} });//[batch, derive, latent]=[batch, derive, featsz]*[featsz latent]
					auto h = hidden->dot(_whh, { {2}, {0} });//[batch, derive, latent] = [batch, derive, latent][latent, latent]
					auto a = *(*x + *h) + *_bh;//[batch, derive, latent]=[batch, derive, latent]+[batch, derive, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//각 커널의 마지막 스텝 히든 적재
				}
			} else if(pos_kern == 2) {//wxh는 커널 스텝별로 가중치를 두고 커널내에서만 위치 분별하고 party단위로 공유하고  
				if(prtbuild) printf("kernel expand 2\n");//whh는 하나만 두어  모두 공유한다.
				Flux *_whh = nullx, *_bh = nullx;
				wxh = flux(tcrAnt, { kernsz, featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				bh = flux(tcrAnt, { 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
				inp = inp->reshape({ -1, n_derive, n_part * kernsz, featsz });//[batch, derive, party * kernel, featsz]
				hiddens = (Flux **)trc->xalloc((n_part * kernsz) * sizeof(Flux *));
				//inp == [batch, derive, party * kernel, featsz]
				auto unstacked_inputs = inp->unstack(2);//[party * kernel](t_length) *[batch, derive, featsz](one - seq),
				auto unstacked_wxh = wxh->unstack(0);//[party * kernel](t_length) *[featsz latent](one - seq),
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, derive, featsz]
					auto _wxh = unstacked_wxh->at(t / kernsz);//[featsz latent]
					if(t % kernsz) hidden = hiddens[t - 1];
					else {//hidden - [batch, derive, latent],각 party의 시작 t에서 초기값 히든을~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~설정하여 party간에 이전값이 곱해지지않게 한다.
					}
					auto x = input_t->dot(_wxh, { {2}, {0} });//[batch, derive, latent]=[batch, derive, featsz]*[featsz latent]
					auto h = hidden->dot(whh, { {2}, {0} });//[batch, derive, latent] = [batch, derive, latent][latent, latent]
					auto a = *(*x + *h) + *bh;//[batch, derive, latent]=[batch, derive, latent]+[batch, derive, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//각 커널의 마지막 스텝 히든 적재
				}
			} else {//wxh는 모든 커널 스텝별로 가중치를 두어 위치 분별하고 whh는 하나만 두어 
				if(prtbuild) printf("kernel expand 1\n");//모두 공유한다.
				Flux *_whh = nullx, *_bh = nullx;
				wxh = flux(tcrAnt, { n_part * kernsz, featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				bh = flux(tcrAnt, { 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
				inp = inp->reshape({ -1, n_derive, n_part * kernsz, featsz });//[batch, derive, party * kernel, featsz]
				hiddens = (Flux **)trc->xalloc((n_part * kernsz) * sizeof(Flux *));
				//inp == [batch, derive, party * kernel, featsz]
				auto unstacked_inputs = inp->unstack(2);//[party * kernel](t_length) *[batch, derive, featsz](one - seq),
				auto unstacked_wxh = wxh->unstack(0);//[party * kernel](t_length) *[featsz latent](one - seq),
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, derive, featsz]
					auto _wxh = unstacked_wxh->at(t);//[featsz latent]
					if(t % kernsz) hidden = hiddens[t - 1];
					else {//hidden - [batch, derive, latent],각 party의 시작 t에서 초기값 히든을~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~설정하여 party간에 이전값이 곱해지지않게 한다.
					}
					auto x = input_t->dot(_wxh, { {2}, {0} });//[batch, derive, latent]=[batch, derive, featsz]*[featsz latent]
					auto h = hidden->dot(whh, { {2}, {0} });//[batch, derive, latent] = [batch, derive, latent][latent, latent]
					auto a = *(*x + *h) + *bh;//[batch, derive, latent]=[batch, derive, latent]+[batch, derive, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//각 커널의 마지막 스텝 히든 적재
				}
			}
			if(dual == 0) hid_cat.push_back(hiddens[t - 1]);//마지막 스텝 히든 적재
			hidden = concat(&hid_cat, 1);
			if(dual) {//[batch, party * kernel * derive, latent]
				hidden = hidden->reshape({ -1, n_part, kernsz, n_derive, latsz });//[batch, party, kernel, derive, latent]
				if(derivefeat == 1) hidden = hidden->transpose({ 0, 2, 1, 3, 4 });//[batch, kernel, party, derive, latent]
				else if(derivefeat == 0) hidden = hidden->transpose({ 0, 2, 1, 4, 3 });//[batch, kernel, party, latent, derive]
			} else {//[batch, party * derive, latent]
				hidden = hidden->reshape({ -1, n_part, n_derive, latsz });//[batch, party, derive, latent]
			}
		} else {//가중치를 party안에서 커널 스텝별로 두어 위치 구분, party단위로 공유
			if(pos_kern == 0) {
				if(prtbuild) printf("kernel expand 0\n");
				inp = inp->reshape({ -1, n_part * n_derive, kernsz, featsz });//[batch, bindbatch(party*derive), kernel, featsz]
				wxh = flux(tcrAnt, { kernsz, featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { kernsz, latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				wh = concat({ wxh, whh }, 1);//[kernel, (featsz+latent) latent]
				bh = flux(tcrAnt, { kernsz, 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				hiddens = (Flux **)trc->xalloc((kernsz + 1) * sizeof(Flux *));
				state = flux(tcrAnt, { inp->fshape[0], n_bind, latsz }, lat_tAnt, constant);
				state->fill(0.0, inp);
				hiddens[kernsz] = state;
				//inp == [batch, bindbatch, kernel, featsz], wh = [kernel, (featsz+latent) latent]
				auto unstacked_inputs = inp->unstack(2);//kernel(t_length) *[batch, bindbatch, featsz](one - seq),
				auto unstacked_wh = wh->unstack(0);//kernel(t_length) *[(featsz+latent) latent](one - seq),
				auto unstacked_bh = bh->unstack(0);
				for(; t < unstacked_inputs->size(); t++) {//input_t: [batch, bindbatch, featsz](t - seq)
					auto input_t = unstacked_inputs->at(t);//[batch, bindbatch, featsz]
					//auto _Wxh = unstacked_Wxh->at(t);
					//auto _Whh = unstacked_Whh->at(t);
					auto _wh = unstacked_wh->at(t);//[(featsz + latent) latent]
					auto _bh = unstacked_bh->at(t);
					hidden = (t == 0 ? hiddens[kernsz] : hiddens[t - 1]);//[batch, bindbatch, latent]
					//_wh = concat({ _Wxh, _Whh }, 0);//[(featsz + latent) latent] = [featsz latent] + [latent, latent]
					auto concat_x = concat({ input_t, hidden }, 2);//[batch, bindbatch, (featsz + latent)] = [batch, bindbatch, featsz](1) + [batch, bindbatch, latent]
					auto a = *concat_x->dot(_wh, { {2}, {0} }) + *_bh;//[batch, bindbatch, latent] = [batch, bindbatch, (featsz + latent)][(featsz + latent), latent], 이하 bindbatch사이즈는 실제로 바인드 사이즈이다.
					//if(prtbuild) { printf("enocde gate[batch, bindbatch, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, bindbatch, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
				}
			} else if(pos_kern == -1) {//모든 party가 커널을 공유하여 wxh는 커널 스텝별로 가중치를 
				if(prtbuild) printf("kernel expand -1\n");//두어 위치 분별하고 whh는 하나만 두어 공유한다.
				inp = inp->reshape({ -1, n_part * n_derive, kernsz, featsz });//[batch, bindbatch(party*derive), kernel, featsz]
				wxh = flux(tcrAnt, { kernsz, featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				bh = flux(tcrAnt, { 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				hiddens = (Flux **)trc->xalloc((kernsz + 1) * sizeof(Flux *));
				state = flux(tcrAnt, { inp->fshape[0], n_bind, latsz }, lat_tAnt, constant);
				state->fill(0.0, inp);
				hiddens[kernsz] = state;
				//inp == [batch, bindbatch, kernel, featsz], wxh == [kernel, featsz, latent]
				auto unstacked_inputs = inp->unstack(2);//kernel(t_length) *[batch, bindbatch, featsz](one - seq),
				auto unstacked_wxh = wxh->unstack(0);//kernel(t_length) *[featsz latent](one - seq),
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, bindbatch, featsz]
					auto _wxh = unstacked_wxh->at(t);//[featsz latent]
					hidden = (t == 0 ? hiddens[kernsz] : hiddens[t - 1]);//[batch, bindbatch, latent]
					auto x = input_t->dot(_wxh, { {2}, {0} });//[batch, bindbatch, latent]=[batch, bindbatch, featsz]*[featsz latent]
					auto h = hidden->dot(whh, { {2}, {0} });//[batch, bindbatch, latent] = [batch, bindbatch, latent][latent, latent]
					auto a = *(*x + *h) + *bh;//[batch, bindbatch, latent]=[batch, bindbatch, latent]+[batch, bindbatch, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, bindbatch, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, bindbatch, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
				}
			} else if(pos_kern == -2) {//wxh, whh를 하나만 두어 공유한다.
				if(prtbuild) printf("kernel expand -2\n");
				inp = inp->reshape({ -1, n_part * n_derive, kernsz, featsz });//[batch, bindbatch(party*derive), kernel, featsz]
				wxh = flux(tcrAnt, { featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				bh = flux(tcrAnt, { 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				hiddens = (Flux **)trc->xalloc((kernsz + 1) * sizeof(Flux *));
				state = flux(tcrAnt, { inp->fshape[0], n_bind, latsz }, lat_tAnt, constant);
				state->fill(0.0, inp);
				hiddens[kernsz] = state;
				//inp == [batch, bindbatch, kernel, featsz], wxh == [featsz, latent]
				auto unstacked_inputs = inp->unstack(2);//kernel(t_length) *[batch, bindbatch, featsz](one - seq),
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, bindbatch, featsz]
					hidden = (t == 0 ? hiddens[kernsz] : hiddens[t - 1]);//[batch, bindbatch, latent]
					auto x = input_t->dot(wxh, { {2}, {0} });//[batch, bindbatch, latent]=[batch, bindbatch, featsz]*[featsz latent]
					auto h = hidden->dot(whh, { {2}, {0} });//[batch, bindbatch, latent] = [batch, bindbatch, latent][latent, latent]
					auto a = *(*x + *h) + *bh;//[batch, bindbatch, latent]=[batch, bindbatch, latent]+[batch, bindbatch, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, bindbatch, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, bindbatch, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
				}
			} else if(pos_kern == -3) {//wxh를 하나만 두어 공유한다. whh는 derive별로 둔다.
				if(prtbuild) printf("kernel expand -3\n");
				wxh = flux(tcrAnt, { featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { n_derive, latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				bh = flux(tcrAnt, { 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				hiddens = (Flux **)trc->xalloc((kernsz + 1) * sizeof(Flux *));
				state = flux(tcrAnt, { inp->fshape[0], n_part, n_derive, latsz }, lat_tAnt, constant);
				state->fill(0.0, inp);
				hiddens[kernsz] = state;
				//inp == [batch, party, derive, kernel, featsz], wxh == [featsz, latent]
				auto unstacked_inputs = inp->unstack(3);//kernel(t_length) *[batch, party, derive, featsz](one - seq),
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, party, derive, featsz]
					hidden = (t == 0 ? hiddens[kernsz] : hiddens[t - 1]);//[batch, party, derive, latent]
					auto x = input_t->dot(wxh, { {3}, {0} });//[batch, party, derive, latent]=[batch, party, derive, featsz]*[featsz latent]
					auto h = trippledot2(hidden, whh);//[batch, party, derive, latent] = [batch, party, derive, latent][n_derive, latent, latent]
					auto a = *(*x + *h) + *bh;//[batch, party, derive, latent]=[batch, party, derive, latent]+[batch, party, derive, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, party, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, party, derive, latent]
					if(dual) {
						a = a->reshape({ -1, n_part * n_derive, latsz });//[batch, bindbatch, latent]
						hid_cat.push_back(a);
					}
				}
			} else if(pos_kern == -4) {//wxh는 커널 스텝별로 가중치를 두고 커널내에서만 위치 분별하고 
				if(prtbuild) printf("kernel expand -4\n");//party단위로 공유하고. whh는 derive별로 둔다.
				wxh = flux(tcrAnt, { kernsz, featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { n_derive, latsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				bh = flux(tcrAnt, { 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				hiddens = (Flux **)trc->xalloc((kernsz + 1) * sizeof(Flux *));
				state = flux(tcrAnt, { inp->fshape[0], n_part, n_derive, latsz }, lat_tAnt, constant);
				state->fill(0.0, inp);
				hiddens[kernsz] = state;
				//inp == [batch, party, derive, kernel, featsz], wxh == [featsz, latent]
				auto unstacked_inputs = inp->unstack(3);//kernel(t_length) *[batch, party, derive, featsz](one - seq),
				auto unstacked_wxh = wxh->unstack(0);//kernel(t_length) *[featsz latent](one - seq),
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, party, derive, featsz]
					auto _wxh = unstacked_wxh->at(t);//[featsz latent]
					hidden = (t == 0 ? hiddens[kernsz] : hiddens[t - 1]);//[batch, party, derive, latent]
					auto x = input_t->dot(_wxh, { {3}, {0} });//[batch, party, derive, latent]=[batch, party, derive, featsz]*[featsz latent]
					auto h = trippledot2(hidden, whh);//[batch, party, derive, latent] = [batch, party, derive, latent][n_derive, latent, latent]
					auto a = *(*x + *h) + *bh;//[batch, party, derive, latent]=[batch, party, derive, latent]+[batch, party, derive, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, party, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, party, derive, latent]
					if(dual) {
						a = a->reshape({ -1, n_part * n_derive, latsz });//[batch, bindbatch, latent]
						hid_cat.push_back(a);
					}
				}
			} else if(pos_kern == -5) {//wxh를 하나만 두어 공유한다. whh는 derive별로 둔다. 제너릭 호출할때 latsz를 
				if(prtbuild) printf("kernel expand -5\n");//n_derive 사이즈로 줘야하고 dual encoder에서 리지주얼
				latsz = 1;									//될수없다. 결과 좋지 않음, 테스트 용
				wxh = flux(tcrAnt, { featsz, latsz }, lat_tAnt, trainable, Initializer::xavier, "wxh", -1);
				whh = flux(tcrAnt, { n_derive, latsz }, lat_tAnt, trainable, Initializer::xavier, "whh", iname);
				bh = flux(tcrAnt, { 1, latsz }, lat_tAnt, trainable, Initializer::xavier, "bh", iname);
				hiddens = (Flux **)trc->xalloc((kernsz + latsz) * sizeof(Flux *));
				state = flux(tcrAnt, { inp->fshape[0], n_part, n_derive, latsz }, lat_tAnt, constant);
				state->fill(0.0, inp);
				hiddens[kernsz] = state;
				//inp == [batch, party, derive, kernel, featsz], wxh == [featsz, latent]
				auto unstacked_inputs = inp->unstack(3);//kernel(t_length) *[batch, party, derive, featsz](one - seq),
				for(; t < unstacked_inputs->size(); t++) {
					auto input_t = unstacked_inputs->at(t);//[batch, party, derive, featsz]
					hidden = (t == 0 ? hiddens[kernsz] : hiddens[t - 1]);//[batch, party, derive, latent]
					auto x = input_t->dot(wxh, { {3}, {0} });//[batch, party, derive, 1]=[batch, party, derive, featsz]*[featsz, 1]
					auto h = *hidden * *whh; //[batch, party, derive, 1] = [batch, party, derive, 1] * [n_derive, 1]
					auto a = *(*x + *h) + *bh;//[batch, party, derive, 1]=[batch, party, derive, 1]+[batch, party, derive, 1]+[1 1]
					//if(prtbuild) { printf("enocde gate[batch, party, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, party, derive, 1]
					if(dual) {
						a = a->reshape({ -1, n_part * n_derive, latsz });//[batch, bindbatch, 1]
						hid_cat.push_back(a);
					}
				}
			} else {
				hiddens = nullx;
				throwFault(-1, "non def kerneling case\n");
			}
			if(dual) {
				hidden = concat(&hid_cat, 1);//[batch, kernel * bindbatch(party*derive), latent]
				hidden = hidden->reshape({ -1, kernsz, n_part, n_derive, latsz });//[batch, kernel, party, derive, latent]
				if(derivefeat == 0) hidden = hidden->transpose({ 0, 1, 2, 4, 3 });//[batch, kernel, party, latent, derive]
				else if(derivefeat == 2) hidden = hidden->transpose({ 0, 2, 1, 3, 4 });//[batch, party, kernel, derive, latent]
			} else {//[batch, bindbatch(party*derive), latent]
				hidden = hiddens[t - 1];//last[batch, bindbatch, latent]
				hidden = hidden->reshape({ -1, n_part, n_derive, latsz });//[batch, party, derive, latent]
			}
		}
		return hidden;
	}
	//ni_part - 0보다 적으면 일반 인코더, 디코더에서 호출, 0 이상이면 듀얼인코더에서 호출
	//kernsz - 듀얼 인코더에서 커널사이즈(8)의 모든 스텝에서 출력할때 의미, 듀얼 인코더가 아닌 곳에서 
	//호출되면 커널의 마지막 스텝만 출력되므로 이 값은 1로 호출된다.
	Flux *_deriveReduce(Flux *hidden, intt rderivet, intt kernsz, intt n_part, intt latsz,
		intt n_derive, intt ni_part, sytet attend, intt cross_out, const bytet *name = "derive_reduce")
	{
		Trace *trc = TRACER(tcrAnt);
		Flux *wcross, *wcross_b = nullx;
		//hidden - if(trc->derivefeat == 2) [batch, party, kernel, derive, latent]
		//			else [batch, kernel, party, latent, derive]
		//ni_part = 4;//예)psz=8, ni_part=4, batch=1, latent=32, cross_out=1
		if(trc->derivefeat == 2) {
			if(cross_out != 1) throwFault(-1, "out size must 1\n");
			if(trc->poskern == -5) hidden = hidden->squeeze(4);
			else if(trc->directMax) hidden = hidden->vmax(hidden->fdim - 2);//[batch, party, kernel, latsz]
			else {
				trc->namescope(name);
				hidden = fillter_dot(hidden, latsz);//[batch, party, kernel, latsz]
				trc->endscope();
				hidden = hidden->actf(actfAnt);
			}
			return hidden;
		} 
		trc->namescope(name);
		if(trc->reducea) rderivet = 0;
		if(rderivet == 3) {
			hidden = hidden->transpose({ 0, 1, 3, 2, 4 });//[[batch, kfinal, latent, party, derive]
			hidden = hidden->reshape({ -1, latsz, n_part*n_derive });//[batch, 1k*latent, party*derive]
			wcross = flux(tcrAnt, { n_part*n_derive, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
			if(prtbuild) { printf("derive_reducer derive reduce 3 [batch, 1k*latent, party*derive]\n"); hidden->shape(); }
			hidden = *hidden->dot(wcross, { {2}, {0} }) + *wcross_b;//[batch, latent, cross_out]
			hidden = hidden->transpose({ 0, 2, 1 });//[batch, cross_out, latent]
		} else if(rderivet == 2) {//61.derive reduce를 매 커널 타임별 가중치를 곱하여 한다.
			wcross = flux(tcrAnt, { kernsz*n_part, n_derive, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { kernsz*n_part, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
			if(ni_part >= 0) hidden = hidden->reshape({ -1, kernsz * n_part, latsz, n_derive });//[batch, kernel * party, latent, derive]
			if(prtbuild) { printf("derive_reducer derive reduce 2 [batch, kernel(%d) * party, latent, derive]\n", kernsz); hidden->shape(); }
			//[batch, kernel * party, latent, cross_out] = [batch, kernel * party, latent, derive] * [kernel * party, derive, cross_out] 
			hidden = stridedot(hidden, wcross, wcross_b);//[1,8*8,32,1]
			if(ni_part >= 0) hidden = hidden->reshape({ -1, kernsz, n_part, latsz, cross_out });//[batch, kernel, party, latent, cross_out]
		} else if(rderivet) {//TRUE RIGHT.derive reduce를 party별 가중치를 곱하여 한다.
			if(attend) {
				if(ni_part >= 0) {
					hidden = hidden->reshape({ -1, kernsz * n_part, latsz, n_derive });
					hidden = dualatten_dot(hidden, cross_out, kernsz, n_part, ni_part);
					if(ni_part >= 0) hidden = hidden->reshape({ -1, kernsz, n_part, latsz, cross_out });
				} else {
					//[batch, party, latent, cross_out] = [batch, party, latent, derive]
					hidden = attention_dot(hidden, cross_out, n_part);
				}
			} else {//TRUE RIGHT.
				wcross = flux(tcrAnt, { n_part, n_derive, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
				wcross_b = flux(tcrAnt, { n_part, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
				if(trc->poskern == 0) hidden = hidden->reshape({ -1, kernsz, n_part, latsz, n_derive });//[batch, kernel, party, latent, derive]
				if(prtbuild) { printf("derive_reducer derive reduce 1 [batch, kernel(%d), party, latent, derive]\n", kernsz); hidden->shape(); }
				//[batch, kernel, party, latent, cross_out] = [batch, kernel, party, latent, derive] * [party, derive, cross_out] 
				hidden = dual_stridedot(hidden, wcross, wcross_b);//[1,8,8,32,1]
			}
		} else {//derive reduce를 모든 스텝에서 가중치를 공유하여 곱한다.
			if(trc->poskern == 0 && ni_part >= 0) hidden = hidden->reshape({ -1, kernsz, n_part, latsz, n_derive });//[batch, kernel, party, latent, derive]
			if(prtbuild) { printf("derive_reducer derive reduce 0 attend %d [batch, kernel(%d), party, latent, derive]\n", attend, kernsz); hidden->shape(); }
			if(attend) {//derivefeat가 설정됐으면 latent 와 derive는 사로 바뀌 상태이다.
				hidden = hidden->reshape({ -1, latsz, n_derive });//[batch*kernel*party, latent, derive]
				hidden = attention_layer(hidden, cross_out);//[batch*kernel*party, latent, cross_out]
				if(ni_part >= 0) hidden = hidden->reshape({ -1, kernsz, n_part, latsz, cross_out });//[batch, kernel, party, latent, cross_out]
			} else {
				intt idot = (ni_part >= 0 ? 4 : 3);
				wcross = flux(tcrAnt, { n_derive, cross_out }, lat_tAnt, (trc->reducea ? constant : trainable), Initializer::xavier, "wcross");
				if(trc->reducea == 0) wcross_b = flux(tcrAnt, { cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
				//[batch, kernel, party, latent, cross_out] = [batch, kernel, party, latent, derive] * [derive, cross_out] 
				if(trc->reducea) {//latent별로 derive를 더한다.
					wcross->fill(1.0);
					hidden = hidden->dot(wcross, { {idot}, {0} });
				} else hidden = *hidden->dot(wcross, { {idot}, {0} }) + *wcross_b;//[1,8,8,32,1]
			}
		}
		trc->endscope();
		return hidden->actf(actfAnt);//[batch, kernel, party, latent, cross_out]
	}
	Flux *deriveReduce(Flux *hidden, intt rderivet, intt kernsz, intt n_part, intt latsz,
		intt n_derive, intt ni_part, sytet attend, intt cross_out, const bytet *name = "derive_reduce")
	{
		Trace *trc = TRACER(tcrAnt);
		bytet s[1024];
		intt nout;

		do {
			if(trc->udreduce) {
				if(n_derive / 2 > cross_out && n_derive / 2 > trc->udreduce) {
					for(nout = 1; nout < n_derive / 2; nout *= 2);
				} else nout = cross_out;
			} else nout = cross_out;
			sprintf(s, "%s_%d", name, nout);
			hidden = _deriveReduce(hidden, rderivet, kernsz, n_part, latsz, n_derive,
				ni_part, attend, nout, s);
			if(trc->derivefeat == 2) break;
			n_derive = hidden->fshape[hidden->fdim - 1];
		} while(n_derive > cross_out);

		return hidden;
	}
	intt emaskAlignment(Trace *trc, Flux *&emask, intt &sz_head)
	{
		intt n_derive = emask->fshape[0];//스트라이드 단위 파생 시퀀스 갯수
		if(trc->multihead == 1) sz_head = 1;
		else if(trc->multihead > 1) {
			sz_head = (n_derive / trc->multihead);
			if(sz_head <= 1) sz_head = 1;//sz_head가 1이면 n_derive를 1로 등분하게되므로 변화없다.
			else if(n_derive % trc->multihead) {
				sz_head++;
				if(trc->outerMask) {//모라르는 것 제로 패딩
					n_derive = (n_derive / sz_head) * sz_head + sz_head;
					auto a = flux(emask, constant);
					a->resizing4(n_derive - emask->fshape[0]);//emask의 첫째행 원래 크기에서   
					a->fill(0.0);		//n_derive로 증가된 증분 사이즈의 제로패딩 마스크 생성
					emask = concat({ emask, a }, 0);//emask의 첫째행 크기를 n_derive에 맞춤
				} else {//나머지를 버림
					if(sz_head) n_derive = (n_derive / sz_head) * sz_head;
					if(n_derive < emask->fshape[0]) emask->resizing4(n_derive);
				}
			}
		}
		if(prtbuild) { printf("multihead mask: n_derive: %d sz_head: %d outer: %d ", n_derive, sz_head, trc->outerMask); emask->shape(); }
		return n_derive;
	}
#define OnlyFinalEncodingAttentionOpt(trc, n_part, spot_att) (spot_att && n_part == 1)
#define AttentionOrOnlyFinalEncodingAttentionOpt(trc, n_part, spot_att) (trc->attsign && (trc->spotOpt != 3 || OnlyFinalEncodingAttentionOpt(trc, n_part, spot_att)))
	Flux *encodeGate(Flux *inp, intt outsz, intt indiscret, intt embedim, sytet interv = -1, intt kernsz = -1,
		intt latsz = -1, sytet dot_t = -1, const bytet *name = "anet_encode")
	{
		Trace *trc = TRACER(tcrAnt);
		bool spot_att = (latsz == 0 || latsz < -1 ? 0 : 1);
		if(entrygate == nullx) entrygate = inp;
		if(interv < 0) interv = intervAnt;
		if(kernsz < 0) kernsz = kernszAnt;
		if(dot_t < 0) dot_t = dot_tAnt;
		if(latsz < -1) latsz *= -1;
		else if(latsz <= 0) latsz = latszAnt;
		if(cmodAnt < 0) {//첫번째 인코딩이면 설정되고 이후에는 이 설정대로 
			if(outsz > 0 && inp->fshape[1] > outsz) cmodAnt = C_ZZ;//입력 시퀀스 길이보다 출력 시퀀스 길이가 적으면 시퀀스 축소 압축 실행.
			else {//양측이 같은 경우이거나 outsz가 음수이면 무조건 시퀀스 축소없이 한번에 타겟 연결
				if(outsz < 0) outsz *= -1;
				cmodAnt = C_XT;
			}
		}
		lat_tAnt = inp->qType;
		if(prtbuild) {printf("encode gate: "); inp->shape();}
		trc->namescope(name);
		intt seqsz = inp->fshape[1], featsz = inp->fshape[2], n_derive, sz_head = 0;
		Flux *dexpend, *residual = inp;
		intt width = (kernsz < widthAnt ? kernsz : widthAnt);
		if(kernsz < 3) throwFault(-1, "seq size too small\n");//3보다 작으면 컴비네이션에서 조합되지 않을수있다.
		if(trc->spotOpt == 1 && seqsz == kernsz) latsz = seqsz;
		//가급적이면 시퀀스 길이를 커널 사이즈/스트라이드의 배수로 하여 두번째 케이스로 실행한다. 시퀀스와 커널이 정합되지 
		//않으면(스트라이드에 따라 아닐수도 있지먄, 예를 들어 시퀀스32, 커널8, 스트라이드6이면 스트라이드 4번째(옵셋 24)에서 
		//멈추어 제로패딩하지 않음) 제로페딩이 설정되므로 학습 효율이 떨어질것.
		floatt r_contig = (width < 8 && rExContigAnt < 0.7 ? 0.7 : rExContigAnt);//폭이작고 rate가 적으면 조합갯수가 0가 되므로 이때는 비율을 높힌다.
		//floatt r_contig = (width < 8 ? 1 : rExContigAnt);
		if(trc->ebatch_t == 2) {
			Flux *a;
			a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			if(a->end_p() == nullx) throwFault(-1, "mask memory alloc fail\n");
			a->fill(1.0);
			auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - 스트라이드 단위 파생 시퀀스 갯수
			//expand_mask->shape();
			//expand_mask->printo(1,2);
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			dexpend = expand_batch2(inp, expand_mask, strideAnt, kernsz, zpadAnt, outsz);//배치의 모든 시퀀스들의 파생 조합 시퀀스 생성
			//dexpend->printo(1, 2);
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, strideAnt, r_contig, zpadAnt, 0, 0);//체크용 코드
			//if(a->fshape[0] != dexpend->fshape[1]) throwFault(-1, "check error\n");
		} else if(trc->ebatch_t || outsz % kernsz || seqsz % kernsz ||
			((floatt)kernsz / (floatt)strideAnt != 1.0) && ((floatt)kernsz / (floatt)strideAnt != 2.0)) {
			//컴비네이션 함수에 의한 stride는 역전파 할수 없으므로 첫번째 encode에만 사용한다. 커널사이즈를 widthAnt보다
			//크게 하면 조합이 너무 많아지므로 이 수이하로 제한하다.
			dexpend = trc->tcr_combination2(inp, kernsz, strideAnt, r_contig, zpadAnt, 1);
			n_derive = tcrAnt->tcr_reserve;
			if(dexpend->fshape[1] % n_derive) throwFault(-1, "not aligned\n");//체크용 코드
			//dexpend->printo(1, 2);//[batch, bindbatch(devide_seq+derive), width, feat]
			//dexpend->shape();
		} else {
			auto *a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			a->fill(1.0);
			auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - 스트라이드 단위 파생 시퀀스 갯수
			//expand_mask->shape();
			//expand_mask->printo();
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			dexpend = expand_batch(inp, expand_mask, interv, kernsz);//배치의 모든 시퀀스들의 파생 조합 시퀀스 생성
			//dexpend->printo(1, 2);
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, stride, r_contig, zpadAnt, 0, 0);//체크용 코드
			//if(a->fshape[0] != dexpend->fshape[1]) throwFault(-1, "check error\n");
		}//depand - [batch, bindbatch(party*derive), kernel, featsz]
		intt n_bind = dexpend->fshape[1];//한개 시퀀스의 총 파생 조합 시붠스 갯수
		intt n_part = n_bind / n_derive;//transform stride단위가 시퀀스 길이에 정수배로 맞아 떨어지게 해야함, 아니면 transform에서 zero padding설정해야함.
		intt n_reduce = outsz / n_part;//커널 윈도우내 파생 조합 시퀀스 단위별로 축소되는 출력 갯수
		if(n_reduce == 0) {//출력 사이즈가 분할 수(입력길이를 커널길이로 나눈 갯수) 보다 적으면, 즉 쵀대압축 출력갯수보다 적으면
			if(n_part % outsz == 0) {//출력 사이즈가 분할 수에 정수배로 맞으면, 예로 분할이 4, 출력이 2이면
				n_derive *= (n_part / outsz);//분할 갯수를 출력 사이즈의 배수 단위로 줄이고 커널단위 파생조합 시퀀스의 갯수를
				n_part = outsz;//위 비율만큼 늘려서 ㄱ)에서 출력사이즈 변동없이 분할 갯수 이하로 할수있게 한다.
			}//분할갯수와 조합갯수를 조정할수없어 ㄱ)에서 출력사이즈가 분할 갯수에 맞게 늘어난다.
			n_reduce = 1;//최소 분할당 1개 이하로 축소될 수 없다.
		}
		intt tarsz = outsz;
		outsz = n_part * n_reduce;//ㄱ.
		if(prtbuild) printf("encode gate seq sz: %d feat sz: %d tsz: %d stride: %d kernel: %d derive: %d party: %d reducing: %d outsz: %d\n", seqsz, featsz, tarsz, strideAnt, kernsz, n_derive, n_part, n_reduce, outsz);
		if(tcrAnt->dbgStep == 1) dexpend = dexpend->bypass("111\n");
		if(indiscret > 0) {//입력 임베딩 #inp == [batch, bindbatch, kernel, 1]
			if(dexpend->fshape[dexpend->fdim - 1] == 1) inp = dexpend->squeeze(dexpend->fdim - 1);//[batch, bindbatch(party*derive), kernel]//tf.reshape(self.input_data, [-1, self.input_data.shape[1], self.input_data.shape[2]])
			else inp = dexpend;//[batch, bindbatch(party*derive), kernel, featsz]
			inp = embedding(inp, indiscret, embedim, 1);//[batch, bindbatch, kernel, featsz(embedim)]
			if(prtbuild) printf("embedding inp [batch, bindbatch, kernel, featsz(embedim)] "); inp->shape();
			//입력 피쳐가 복수차원이면 각 피쳐를 임베딩후에 피쳐를 한개로 합한다.
			if(inp->fdim > 4) inp = inp->reshape({ inp->fshape[0], inp->fshape[1], inp->fshape[2], -1 });
			featsz = inp->fshape[inp->fdim - 1];//embedim
			//inp = inp->bypass("111-2\n");
			//if(featsz != embedim) throwFault(-1, "embed size\n");
			if(trc->positional > 0) {
				if(trc->positional == 1) trc->positional = 0;//처음 한번만 포지셔널, 음수이면 이하 리지주얼 블럭 계속 포지셔널
				inp = inp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
				inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
				auto enpos = flux(trc, 3, &inp->fshape[2], inp->qType, constant);//[party, kernel, featsz]
				enpos->sinpos(n_part * kernsz);//[seq(party, kernel), featsz]
				inp = *inp + *enpos;//sinuosid positional, [batch, derive, party, kernel, featsz] = [batch, derive, party, kernel, featsz] + [positional(party, kernel), featsz]
				inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, party, derive, kernel, featsz]
			} else inp = inp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
		} else {//TRUE RIGHT.
			inp = dexpend;//inp == [batch, bindbatch(party*derive), kernel, featsz]
			inp = inp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
		}
		if(trc->ehook) inp = expand_hook(inp, kernsz, latsz, featsz, n_part, n_derive);
		Flux *wcross = nullx, *wcross_b = nullx, *hidden;
		if(sz_head) {//derive를 분할하여 커널링 수행.
			intt nhead = n_derive / sz_head, i = 0;
			auto depart = inp->split(nhead, 2);//[batch, party, derive(nhead*sz_head), kernel, featsz]
			vector<Flux *> hids;
			for(auto iter : *depart) {//[batch, party, sz_head, kernel, featsz]
				auto a = kerneling(iter, 0, trc->derivefeat, kernsz, latsz, featsz, n_part * sz_head, n_part, sz_head, i++);
				hids.push_back(a);//a - [batch, party, sz_head, latent]
			}
			hidden = concat(&hids, 2);//[batch, party, derive(nhead*sz_head), latent]
		} else hidden = kerneling(inp, 0, trc->derivefeat, kernsz, latsz, featsz, n_bind, n_part, n_derive, -1);
		//hidden - [batch, party, derive, latent]
		if(prtbuild) { printf("enocde gate[batch, party, derive, latent]\n"); hidden->shape(); }
		if(trc->spotOpt == 1 && seqsz == kernsz) {
			hidden = hidden->reshape({ -1, n_part*n_derive, latsz });//[batch, bindbatch, latent]
			hidden = hidden->transpose({ 0, 2, 1 });//[batch, latent(seq), bindbatch(reducing)]
			trc->endscope();
			return hidden;
		}
		//이하 시퀀스 단위로 시퀀스내의 바인드 배치를 한개 시퀀스 출력으로 변환
		Flux *outtens, *znode, *encode;
		intt cross_out;
		if(cmodAnt == C_XT) cross_out = outsz;// #한번에 출력 연결
		else cross_out = n_reduce;//잠채코드 추상화 연결
		if(trc->reducea) {
			dot_t = ORTHO_DOT;
			trc->ortho_single = 1;
			trc->attsign = 0;
		}
		if(AttentionOrOnlyFinalEncodingAttentionOpt(trc, n_part, spot_att)) goto NO_WEIGHT;//어텐션 수행이면 내부에서 가중치 정의하므로 여기 가중치 필요없다.
		if(dot_t == ORTHO_DOT || dot_t == STRIDE_DOT || trc->derivefeat == 2) goto NO_WEIGHT;
		if(dot_t == TRIPPLE_DOT) {//분할 단위로 가중치 공유.(커널 사이즈내에서만 포지션 구분)
			wcross = flux(tcrAnt, { n_derive, latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { n_derive, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		} else if(dot_t == COUPLE_DOT) {//모든 파생 조합별로 개별 가중치(포시션 구분 의미)
CDOT_LA:;	wcross = flux(tcrAnt, { n_bind, latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { n_bind, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		} else {//TENSOR_DOT, 모든 파생 조합이 가중치 공유(포지션 구분 되지 않음)
			wcross = flux(tcrAnt, { latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		}
NO_WEIGHT:;
		if(dot_t == ORTHO_DOT && trc->derivefeat != 2) {//히든(차원)이 컴볼루션(CNN)의 필터(갯수)에 대응됨. derive는 feature map
			if(trc->spotOpt == 2 && n_part == 1) {//마지막 커널사이즈 길이를 인코딩하는 것이고 개별옵션이면 히든별로 derive를 곱한다.
				hidden = hidden->reshape({ -1, n_part*n_derive, latsz });//[batch, bindbatch, latent]
				hidden = hidden->transpose({ 0, 2, 1 });//[batch, latent, bindbatch(derive)]
				n_derive = hidden->fshape[1];
				goto CDOT_LA2;
			}
			hidden = hidden->transpose({ 0, 1, 3, 2 });//[batch, party, latent, derive]
			if(AttentionOrOnlyFinalEncodingAttentionOpt(trc, n_part, spot_att)) {
				//[batch, party, latent, cross_out] = [batch, party, latent, derive]
				outtens = deriveReduce(hidden, 1, 1, n_part, latsz, n_derive, -1, 1, cross_out);
				if(prtbuild) { printf("enocde ortho attention gate[batch, party, latent, cross_out]\n"); outtens->shape(); }
			} else if(trc->ortho_single) {//분할간에 가중치 공유, 스트라이간에 시퀀스 없음, 입력길이8->출력1을 분할 갯수만큼 여러번 호출하는 것을 한꺼번에 실행하는 것과 같음
				//[batch, party, latent, cross_out] = [batch, party, latent, derive] * [derive, cross_out] 
				outtens = deriveReduce(hidden, 0, 1, n_part, latsz, n_derive, -1, 0, cross_out);
				if(prtbuild) { printf("enocde ortho stride gate[batch, party, latent, cross_out]\n"); outtens->shape(); }
			} else {
				//[batch, party, latent, cross_out] = [batch, party, latent, derive] * [party, derive, cross_out] 
				outtens = deriveReduce(hidden, 2, 1, n_part, latsz, n_derive, -1, 0, cross_out);
				if(prtbuild) { printf("enocde ortho stride gate[batch, party, latent, cross_out]\n"); outtens->shape(); }
			}
			/*if(cmodAnt == C_XT) {//ageg.[batch, bindbatch(party*latent), cross_out] = [batch, party, latent, cross_out]
				outtens = outtens->reshape({ -1, party * outtens->fshape[2], outtens->fshape[3] });
				outtens = outtens->actf(actfAnt);
				trc->endscope();
				return outtens;//[batch, bindbatch, outsz]
			}*/
			znode = outtens;
		} else if(dot_t == STRIDE_DOT || trc->derivefeat == 2) {//TRUE RIGHT.
			if(trc->spotOpt == 2 && n_part == 1 && trc->derivefeat != 2) {
				hidden = hidden->reshape({ -1, n_part*n_derive, latsz });//[batch, bindbatch, latent]
				goto CDOT_LA2;//마지막 커널사이즈 길이를 인코딩하는 것이고 개별옵션이면 derive별로 히든을 곱한다.
			}
			if(AttentionOrOnlyFinalEncodingAttentionOpt(trc, n_part, spot_att)) {
				//[batch, party, derive, cross_out] = [batch, party, derive, latent]
				outtens = deriveReduce(hidden, trc->ortho_single ? 0 : 1, 1, n_part, latsz, n_derive, -1, 1, cross_out);
				if(prtbuild) { printf("enocde stride attention gate[batch, party, derive, cross_out]\n"); outtens->shape(); }
			} else {//TRUE RIGHT.
				//[batch, party, derive, cross_out] = [batch, party, derive, latent] * [party, latent, cross_out] 
				outtens = deriveReduce(hidden, 2, 1, n_part, latsz, n_derive, -1, 0, cross_out);
				if(prtbuild) { printf("enocde stride gate[batch, party, derive, cross_out]\n"); outtens->shape(); }
			}
			/*if(cmodAnt == C_XT) {//ageg.[batch, bindbatch(party*derive), cross_out] = [batch, party, derive, cross_out]
				outtens = outtens->reshape({ -1, party * derive, outtens->fshape[3] });
				outtens = outtens->actf(actfAnt);
				trc->endscope();
				return outtens;//[batch, bindbatch, outsz]
			}*/
			znode = outtens;
		} else if(dot_t == TRIPPLE_DOT) {
			//[batch, party, derive, cross_out] = [batch, party, derive, latent] * [derive, latent, cross_out] 
			outtens = trippledot(hidden, wcross, wcross_b);
			/*if(cmodAnt == C_XT) {//ageg.[batch, bindbatch(party*derive), cross_out] = [batch, party, derive, cross_out]
				outtens = outtens->reshape({ -1, party * derive, outtens->fshape[3] });
				outtens = outtens->actf(actfAnt);
				trc->endscope();
				return outtens;//[batch, bindbatch, outsz]
			}*/
			znode = outtens->actf(actfAnt);
		} else if(dot_t == COUPLE_DOT) {
			hidden = hidden->reshape({ -1, n_part*n_derive, latsz });//[batch, bindbatch, latent]
CDOT_LA2:;	//ORTHO_DOT에서 온거면[batch, latent, cross_out] = [batch, latent, bindbatch] * [latent, bindbatch, cross_out]
			//[batch, bindbatch, cross_out] = [batch, bindbatch, latent] * [bindbatch, latent, cross_out]
			outtens = coupledot(hidden, wcross, wcross_b);
			if(prtbuild) { printf("enocde couple gate[batch, bindbatch, cross_out]\n"); outtens->shape(); }
			/*if(cmodAnt == C_XT) {//ageg.
				outtens = outtens->actf(actfAnt);
				trc->endscope();
				return outtens;//[batch, bindbatch, outsz]
			}*/
			outtens = outtens->reshape({ -1, n_part, n_derive, outtens->fshape[2] });//[batch, party, derive, cross_out]
			znode = outtens->actf(actfAnt);
		} else {
			hidden = hidden->reshape({ -1, n_part*n_derive, latsz });//[batch, bindbatch, latent]
			if(AttentionOrOnlyFinalEncodingAttentionOpt(trc, n_part, spot_att)) {
				//[batch, bindbatch, cross_out] = [batch, bindbatch, latent]
				outtens = attention_dot(hidden, cross_out, 1);
				if(prtbuild) { printf("enocde tensor attention gate[batch, bindbatch, cross_out]\n"); outtens->shape(); }
			} else {
				//[batch, bindbatch, cross_out] = [batch, bindbatch, latent] * [latent, cross_out]
				outtens = (*hidden->dot(wcross, { {2}, {0} }) + *wcross_b);
				if(prtbuild) { printf("enocde tensor gate[batch, bindbatch, cross_out]\n"); outtens->shape(); }
			}
			/*if(cmodAnt == C_XT) {//ageg.
				outtens = outtens->actf(actfAnt);
				trc->endscope();
				return outtens;//[batch, bindbatch, outsz]
			}*/
			outtens = outtens->reshape({ -1, n_part, n_derive, outtens->fshape[2] });//[batch, party, derive, cross_out]
			znode = outtens->actf(actfAnt);
		}
		if(trc->derivefeat != 2) {
			znode = znode->transpose({ 0, 1, 3, 2 });//[batch, party, cross_out(reducing), derive/latent(ORTHO_DOT)]
			if(prtbuild) { printf("enocde gate[batch, party, cross_out(reducing), derive/latent]\n"); znode->shape(); }
			encode = znode->reshape({ -1, outsz, znode->fshape[3] });//[batch, outsz(party*reducing), derive/latent(ORTHO_DOT)], 잠재코드는 시퀀스 순서대로 정렬
		} else encode = znode;//[batch, party, latent]
		if(n_part == 1 && dot_t == ORTHO_DOT && encode->fshape[1] == residual->fshape[1]) encode = *encode + *residual;
		encodeOut = encode;//stratus에서 사용
		if(trc->layer_norm == 2) {
			encode = encode->layer_normal();
			encode = encode->actf(actfAnt);
		}
		trc->endscope();
		if(tcrAnt->dbgStep == 1) encode = encode->bypass("333\n");
		if(prtbuild) { printf("enocde gate out: [batch, outsz(party*reducing), derive/latent]\n"); encode->shape(); }
		return encode;
	}
	Flux *decodeGate(Flux *inp, intt outsz, intt kernsz = -1, sytet dot_t = -1, sytet elastic = -1, const bytet *name = "anet_decode")
	{
		Trace *trc = TRACER(tcrAnt);
		Flux *decode;
		bool decatt = 0;
		if(kernsz < 0) kernsz = kernszAnt;
		if(trc->dot_tDec >= 0) dot_t = trc->dot_tDec;
		else if(dot_t < 0) dot_t = dot_tAnt;
		if(elastic < 0) elastic = elasticAnt;
		//inp = inp->bypass("bbb\n");
		tcrAnt->namescope(name);
		if(cmodAnt == C_XT) {//inp == [batch, bindbatch, cross_out]//ageg)를 실행안하여 출력 
			/*if(elastic) {//depricate	//시퀀스 길이[batch, outsz, lattent]로 바로 출력 하기때문에
				intt n_bind = inp->fshape[1];	//디코더는 스킵한다.
				intt cross_out = inp->fshape[2];
				decode = inp->layer_dense(cross_out / 2, actfAnt);//[batch, bindbatch, cross_out]
				decode = decode->transpose({ 0, 2, 1 });//[batch, cross_out, bindbatch]
				decode = decode->layer_dense(n_bind / 2, actfAnt);
				decode = decode->transpose({ 0, 2, 1 });//[batch, bindbatch, cross_out]
				decode = decode->layer_dense(cross_out, actfAnt);
				decode = decode->transpose({ 0, 2, 1 });//[batch, cross_out, bindbatch]
				decode = decode->layer_dense(n_bind, actfAnt);//[batch, cross_out(y_sz), bindbatch(latent)]
			} else decode = inp->transpose({ 0, 2, 1 });//[batch, cross_out(y_sz), bindbatch(latent)]*/
			decode = inp;
			if(prtbuild) { printf("decode gate[batch, cross_out(y_sz), bindbatch(latent)]\n"); decode->shape(); }
		} else {//인코딩 시퀀스 사이즈를 디코딩 시퀀스 사이즈로 확대, inp == [batch, insz(party*reducing), derive/latent(ORTHO_DOT)]
			intt insz = inp->fshape[1], nhead;
			intt n_derive = inp->fshape[2];
			Flux *wseq = nullx, *wseq_b = nullx, *znode = nullx;
			if(trc->reducea) dot_t = TENSOR_DOT;
			if(dot_t == REDUCE_DOT) {//inp == [batch, insz(party*reducing), derive/latent(ORTHO_DOT)]
				wseq = flux(tcrAnt, { n_derive, 1 }, lat_tAnt, trainable, Initializer::xavier, "wseq");
				wseq_b = flux(tcrAnt, { 1 }, lat_tAnt, trainable, Initializer::xavier, "wseq_b");
				decode = (*inp->dot(wseq, { {2},{0} }) + *wseq_b)->actf(actfAnt);//[batch, insz, 1] = [batch, insz, hidden]*[hidden, 1]
				Flux *wseq2 = flux(tcrAnt, { 1, outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq2");
				Flux *wseq_b2 = flux(tcrAnt, { outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq_b2");
				decode = (*decode->dot(wseq2, { {2},{0} }) + *wseq_b2)->actf(actfAnt);//[batch, insz, outsz] = [batch, insz, 1]*[1, outsz]
				decode = decode->transpose({ 0, 2, 1 });//[batch, outsz, insz]
				Flux *wseq3 = flux(tcrAnt, { insz, 1 }, lat_tAnt, trainable, Initializer::xavier, "wseq3");
				Flux *wseq_b3 = flux(tcrAnt, { 1 }, lat_tAnt, trainable, Initializer::xavier, "wseq_b3");
				decode = (*decode->dot(wseq3, { {2},{0} }) + *wseq_b3)->actf(actfAnt);//[batch, outsz, 1] = [batch, outsz, insz]*[insz, 1]
				goto LB1;//[batch, outsz, 1(hidden)]
			}
			if(dot_t == TENSOR_DOT || dot_t == TRIPPLE_DOT) {
				if(trc->reducea);
				else if(trc->att_sDec == 1 && insz == kernsz) decatt = 1;
				else if(trc->att_sDec == 2) decatt = 1;
			}
			if(dot_t == TENSOR_DOT || dot_t == TRIPPLE_DOT) {
				if(decatt == 0) {
					wseq = flux(tcrAnt, { insz, outsz }, lat_tAnt, (trc->reducea ? constant : trainable), Initializer::xavier, "wseq");
					if(trc->reducea == 0) wseq_b = flux(tcrAnt, { outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq_b");
				}//else 디코더에서 어텐션 적용 옵션이면 가중치 정의하므로 여기 가중치 필요없다.
			} else if(szmhead) {//TRUE RIGHT.dot_t가 STRIDE_DOT일때만 설정되고 의미있음 - depricate szmhead관련 나중 삭제
				nhead = n_derive / szmhead;
				wseq = flux(tcrAnt, { nhead, insz, outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq");
				wseq_b = flux(tcrAnt, { nhead, outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq_b");
				if(prtbuild) printf("decode gate multi-head\n");
			} else {
				wseq = flux(tcrAnt, { n_derive, insz, outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq");
				wseq_b = flux(tcrAnt, { n_derive, outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq_b");
			}
			znode = inp->transpose({ 0, 2, 1 });//[batch, derive/latent(ORTHO_DOT), insz(party*reducing)], derive간격 interval, interval사이를 추론하게되어 예측결과가 좋음.
			if(dot_t == TENSOR_DOT || dot_t == TRIPPLE_DOT) {
				//[batch, derive, outsz] = [batch, derive/latent(ORTHO_DOT), insz]*[insz. outsz]
				if(decatt) {//ORTHO_DOT가 아니어서 derive이면 현재는 메모리 할당량이 너무 커 할당 에러나고 ORTHO_DOT일때는
					znode = attention_layer(znode, outsz)->actf(actfAnt);//latent가 128을 넘으면 마찬가지로 할당에러 나므로 주의.
				} else if(trc->reducea) {
					wseq->fill(1.0);
					znode = znode->dot(wseq, { {2},{0} })->actf(actfAnt);
				} else znode = (*znode->dot(wseq, { {2},{0} }) + *wseq_b)->actf(actfAnt);
			} else if(szmhead) {//TRUE RIGHT.znode - [batch, derive/latent(ORTHO_DOT), insz(party*reducing)]
				znode = znode->reshape({ -1, nhead, szmhead, insz });//[batch, nhead, szmhead, insz]
				//[batch, nhead, szmhead, outsz] = [batch, nhead, szmhead, insz] * [nhead, insz, outsz]
				znode = stridedot(znode, wseq, wseq_b)->actf(actfAnt);
				znode = znode->reshape({ -1, nhead * szmhead, outsz });//[batch, derive(nhead*szmhead), outsz]
			} else {
				//[batch, derive/latent(ORTHO_DOT), outsz] = [batch, derive/latent(ORTHO_DOT), insz]*[derive/latent(ORTHO_DOT), insz. outsz]
				znode = coupledot(znode, wseq, wseq_b)->actf(actfAnt);
			}
			if(elastic) {
				znode = znode->layer_dense(znode->fshape[2], actfAnt);//[batch, derive(latent), outsz]
				znode = znode->transpose({ 0, 2, 1 });//[batch, outsz, derive]
				decode = znode->layer_dense(znode->fshape[2], actfAnt);//[batch, outsz, derive(latent)]
			} else decode = znode->transpose({ 0, 2, 1 });//[batch, outsz, derive(latent)]
LB1:;
			if(prtbuild) {
				if(decatt) printf("decode gate attention insz: %d outsz: %d [batch, outsz, derive(latent)]\n", insz, outsz);
				else printf("decode gate insz: %d outsz: %d [batch, outsz, derive(latent)]\n", insz, outsz);
				decode->shape();
			}
		}
		tcrAnt->endscope();
		//t_hid = decode->slice({ {}, {0 , 3}, {0, 3} });
		return decode;
	}
	Flux *decodeGate2(Flux *inp, intt outsz, sytet interv = -1, intt kernsz = -1,
		intt latsz = -1, sytet dot_t = -1, const bytet *name = "anet_decode2")
	{
		Trace *trc = TRACER(tcrAnt);
		bool decatt = (trc->att_sDec > 1 ? 1 : 0);
		if(entrygate == nullx) entrygate = inp;
		if(interv < 0) interv = intervAnt;
		if(kernsz < 0) kernsz = kernszAnt;
		if(latsz < 0) latsz = latszAnt;
		if(dot_t < 0) dot_t = dot_tAnt;
		if(dot_t == REDUCE_DOT) throwFault(-1, "decode2 and reduce dot not matching\n");
		//inp == [batch, insz(party*reducing), derive]
		/*nxcicp.완전압축 옵션이라도 디코드2 옵션이면 인코딩에서 완전압축하지 않으므로 필요없다.
		if(inp->fshape[1] == kernsz && kernsz != outsz) { // && finalReduced) {//완전 압축후에 첫번째 디코딩에서만 시퀀스를 전치한다.안그러면
			inp = inp->transpose({ 0, 2, 1 });//[batch, derive, insz(party*reducing)]//완전압축을 동일하게 반복하는 것이 되므로
			if(trc->att_sDec == 1) decatt = 1;
		}*/
		intt seqsz = inp->fshape[1], featsz = inp->fshape[2], n_derive, sz_head = 0;
		Flux *dexpend, *residual = inp;
		intt width = (kernsz < widthAnt ? kernsz : widthAnt);
		if(tcrAnt->dbgStep == 1) inp = inp->bypass("444\n");
		trc->namescope(name);
		floatt r_contig = (width < 8 && rExContigAnt < 0.7 ? 0.7 : rExContigAnt);
		if(trc->ebatch_t == 2 || outsz % kernsz || seqsz % kernsz ||
			((floatt)kernsz / (floatt)strideAnt != 1.0) && ((floatt)kernsz / (floatt)strideAnt != 2.0)) {
			Flux *a;
			a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			a->fill(1.0);
			auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - 스트라이드 단위 파생 시퀀스 갯수
			//expand_mask->shape();
			//expand_mask->printo(1,2);
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			dexpend = expand_batch2(inp, expand_mask, strideAnt, kernsz, zpadAnt, outsz);//배치의 모든 시퀀스들의 파생 조합 시퀀스 생성
			//dexpend->printo(1, 2);
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, strideAnt, r_contig, zpadAnt, 0, 0);//체크용 코드
			//if(a->fshape[0] != dexpend->fshape[1]) throwFault(-1, "check error\n");
		} else {
			auto *a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			a->fill(1.0);
			auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - 스트라이드 단위 파생 시퀀스 갯수
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			dexpend = expand_batch(inp, expand_mask, interv, kernsz);//배치의 모든 시퀀스들의 파생 조합 시퀀스 생성
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, stride, r_contig, zpadAnt, 0, 0);//체크용 코드
			//if(a->fshape[0] != dexpend->fshape[1]) throwFault(-1, "check error\n");
		}
		if(tcrAnt->dbgStep == 1) dexpend = dexpend->bypass("555\n");
		intt n_bind = dexpend->fshape[1];//한개 시퀀스의 총 파행 조합 시붠스 갯수
		intt n_part = n_bind / n_derive;
		intt cross_out = outsz / n_part;//커널 윈도우내 파생 조합 시퀀스 단위별로 축소되는 출력 갯수
		if(cross_out == 0) cross_out = 1;//최소 스트라이드당 1개 이하로 될 수 없다.
		intt tarsz = outsz;
		outsz = n_part * cross_out;
		if(prtbuild) printf("decode gate2 seq sz: %d tsz: %d stride: %d sz_kernel: %d derive: %d party: %d reducing: %d outsz: %d\n", seqsz, tarsz, strideAnt, kernsz, n_derive, n_part, cross_out, outsz);
		//with tf.variable_scope(name) :
		if(trc->spotOpt == 1 && seqsz == kernsz) latsz = outsz;
		inp = dexpend;//inp == [batch, bindbatch(party*derive), kernel, featsz]
		Flux *wcross = nullx, *wcross_b = nullx, *hidden;
		inp = inp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
		if(sz_head) {//derive를 분할하여 커널링 수행.
			intt nhead = n_derive / sz_head, i = 0;
			auto depart = inp->split(nhead, 2);//[batch, party, derive(nhead*sz_head), kernel, featsz]
			vector<Flux *> hids;
			for(auto iter : *depart) {//[batch, party, sz_head, kernel, featsz]
				auto a = kerneling(iter, 0, trc->derivefeat, kernsz, latsz, featsz, n_part * sz_head, n_part, sz_head, i++);
				hids.push_back(a);//a - [batch, party, sz_head, latent]
			}
			hidden = concat(&hids, 2);//[batch, party, derive(nhead*sz_head), latent]
		} else hidden = kerneling(inp, 0, trc->derivefeat, kernsz, latsz, featsz, n_bind, n_part, n_derive, -1);
		//hidden - [batch, party, derive, latent]
		if(prtbuild) { printf("deocde2 gate[batch, party, derive, latent]\n"); hidden->shape(); }
		if(trc->hidPrint == 3) hidden = hidden->bypass("=============3===========");
		if(trc->spotOpt == 1 && seqsz == kernsz) {
			hidden = hidden->reshape({ -1, n_part*n_derive, latsz });//[batch, bindbatch, latent]
			hidden = hidden->transpose({ 0, 2, 1 });//[batch, latent(seq), bindbatch(reducing)]
			trc->endscope();
			return hidden;
		}
		if(trc->reducea) {
			dot_t = ORTHO_DOT;
			trc->ortho_single = 1;
			decatt = 0;
		}
		//이하 시퀀스 단위로 시퀀스내의 바인드 배치를 한개 시퀀스 출력으로 변환
		if(decatt || trc->derivefeat == 2) goto NO_WEIGHT;//어텐션 수행이면 내부에서 가중치 정의하므로 여기 가중치 필요없다.
		if(dot_t == ORTHO_DOT || dot_t == STRIDE_DOT) goto NO_WEIGHT;
		if(dot_t == TRIPPLE_DOT) {//스트라이드 단위로 가중치 공유.(커널 사이즈내에서만 포지션 구분)
			wcross = flux(tcrAnt, { n_derive, latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { n_derive, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		} else if(dot_t == COUPLE_DOT) {//모든 파생 조합별로 개별 가중치(포시션 구분 의미)
CDOT_LA:;	wcross = flux(tcrAnt, { n_bind, latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { n_bind, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		} else {//TENSOR_DOT, 모든 파생 조합이 가중치 공유(포지션 구분 되지 않음)
			wcross = flux(tcrAnt, { latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		}
NO_WEIGHT:;
		Flux *outtens, *znode, *encode;
		if(dot_t == ORTHO_DOT && trc->derivefeat != 2) {
			if(trc->spotOpt == 2 && n_part == 1) {//마지막 커널사이즈 길이를 인코딩하는 것이고 개별옵션이면 히든별로 derive를 곱한다.
				hidden = hidden->reshape({ -1, n_part*n_derive, latsz });//[batch, bindbatch, latent]
				hidden = hidden->transpose({ 0, 2, 1 });//[batch, latent, bindbatch(derive)]
				n_derive = hidden->fshape[1];
				goto CDOT_LA2;
			}
			hidden = hidden->transpose({ 0, 1, 3, 2 });//[batch, party, latent, derive]
			if(decatt) {
				//[batch, party, latent, cross_out] = [batch, party, latent, derive]
				outtens = deriveReduce(hidden, 1, 1, n_part, latsz, n_derive, -1, 1, cross_out);
				if(prtbuild) { printf("deocde2 ortho attention gate[batch, party, latent, cross_out]\n"); outtens->shape(); }
			} else if(trc->ortho_single) {//스트라이드간에 가중치 공유, 스트라이간에 시퀀스 없음
				//[batch, party, latent, cross_out] = [batch, party, latent, derive] * [derive, cross_out] 
				outtens = deriveReduce(hidden, 0, 1, n_part, latsz, n_derive, -1, 0, cross_out);
				if(prtbuild) { printf("deocde2 ortho gate[batch, party, latent, cross_out]\n"); outtens->shape(); }
			} else {
				//[batch, party, latent, cross_out] = [batch, party, latent, derive] * [party, derive, cross_out] 
				outtens = deriveReduce(hidden, 2, 1, n_part, latsz, n_derive, -1, 0, cross_out);
				if(prtbuild) { printf("deocde2 ortho gate[batch, party, latent, cross_out]\n"); outtens->shape(); }
			}
			znode = outtens;
		} else if(dot_t == STRIDE_DOT || trc->derivefeat == 2) {
			if(trc->spotOpt == 2 && n_part == 1 && trc->derivefeat != 2) {
				hidden = hidden->reshape({ -1, n_part*n_derive, latsz });//[batch, bindbatch, latent]
				goto CDOT_LA2;//마지막 커널사이즈 길이를 인코딩하는 것이고 개별옵션이면 derive별로 히든을 곱한다.
			}
			if(decatt) {
				//[batch, party, derive, cross_out] = [batch, party, derive, latent]
				outtens = deriveReduce(hidden, 1, 1, n_part, latsz, n_derive, -1, 1, cross_out);
				if(prtbuild) { printf("deocde2 stride attention gate[batch, party, derive, cross_out]\n"); outtens->shape(); }
			} else {
				//[batch, party, derive, cross_out] = [batch, party, derive, latent] * [party, latent, cross_out] 
				outtens = deriveReduce(hidden, 2, 1, n_part, latsz, n_derive, -1, 0, cross_out);
				if(prtbuild) { printf("deocde2 stride gate[batch, party, derive, cross_out]\n"); outtens->shape(); }
			}
			znode = outtens;
		} else if(dot_t == TRIPPLE_DOT) {
			//[batch, party, derive, cross_out] = [batch, party, derive, latent] * [derive, latent, cross_out] 
			outtens = trippledot(hidden, wcross, wcross_b);
			znode = outtens->actf(actfAnt);
		} else if(dot_t == COUPLE_DOT) {
			hidden = hidden->reshape({ -1, n_part*n_derive, latsz });//[batch, bindbatch, latent]
CDOT_LA2:;	//ORTHO_DOT에서 온거면[batch, latent, cross_out] = [batch, latent, bindbatch] * [latent, bindbatch, cross_out]
			//[batch, bindbatch, cross_out] = [batch, bindbatch, latent] * [bindbatch, latent, cross_out]
			outtens = coupledot(hidden, wcross, wcross_b);
			if(prtbuild) { printf("deocde2 couple gate[batch, bindbatch, cross_out]\n"); outtens->shape(); }
			outtens = outtens->reshape({ -1, n_part, n_derive, outtens->fshape[2] });//[batch, party, derive, cross_out]
			znode = outtens->actf(actfAnt);
		} else {
			hidden = hidden->reshape({ -1, n_part*n_derive, latsz });//[batch, bindbatch, latent]
			if(decatt) {
				//[batch, bindbatch, cross_out] = [batch, bindbatch, latent]
				outtens = attention_dot(hidden, cross_out, 1);
				if(prtbuild) { printf("deocde2 tensor attention gate[batch, bindbatch, cross_out]\n"); outtens->shape(); }
			} else {
				//[batch, bindbatch, cross_out] = [batch, bindbatch, latent] * [latent, cross_out]
				outtens = (*hidden->dot(wcross, { {2}, {0} }) + *wcross_b);
				if(prtbuild) { printf("deocde2 tensor gate[batch, bindbatch, cross_out]\n"); outtens->shape(); }
			}
			outtens = outtens->reshape({ -1, n_part, n_derive, outtens->fshape[2] });//[batch, party, derive, cross_out]
			znode = outtens->actf(actfAnt);
		}
		if(trc->derivefeat != 2) {
			znode = znode->transpose({ 0, 1, 3, 2 });//[batch, party, cross_out(reducing), derive/latent(ORTHO_DOT)]
			if(prtbuild) { printf("deocde2 gate[batch, party, cross_out(reducing), derive/latent]\n"); znode->shape(); }
			encode = znode->reshape({ -1, outsz, znode->fshape[3] });//[batch, outsz(party*reducing), derive/latent(ORTHO_DOT)], 잠재코드는 시퀀스 순서대로 정렬
		} else encode = znode;//[batch, party, latent]
		if(prtbuild) { printf("deocde2 gate[batch, outsz(party*reducing), derive/latent]\n"); encode->shape(); }
		if(seqsz == outsz && dot_t == ORTHO_DOT) {
			encode = *encode * *residual;
			encode = encode->actf(actfAnt);
		}
		trc->endscope();
		if(tcrAnt->dbgStep == 1) encode = encode->bypass("666\n");
		return encode;
	}
	Flux *outGate(Flux *inp, intt ydisc, intt out_feat, Flux *&logits, const bytet *name = "anet_out", bool reuse = 0)//출력 시퀀스의 디멘젼을 타겟 디멘젼으로 변환.
	{
		Flux *pred;

		tcrAnt->namescope(name, reuse);
		if(ydisc) yDiscrete = ydisc;//출력 피쳐가 vocabulary size
		else {
			yDiscrete = 0;
			ydisc = 1;
		}
		auto hidden = inp->fshape[2];//inp == [batch, y_sz, bindbatch(latent)]
		//with tf.variable_scope(name) :
		if(yDiscrete == 0 && hidden == out_feat) logits = inp;
		else {//out_feat은 일반적으로 언어와 같은 경우 토큰아이디 1개 이다.
			intt mo_feat = out_feat * ydisc;//출력 피쳐 갯수가 1보다 크면 그 갯수만큼 피쳐 사이즈 곱
			auto wod = flux(tcrAnt, { hidden, mo_feat }, lat_tAnt, trainable, Initializer::xavier, "wod");//weight output demension
			auto wod_b = flux(tcrAnt, { mo_feat }, lat_tAnt, trainable, Initializer::xavier, "wod_b");
			//yDiscrete이면 [batch, out_seq, (out_feat)*vocab_sz] = [batch, out_seq, latent][latent, (out_feat)*vocab_sz]
			//else			  [batch, out_seq, out_feat] = [batch, out_seq, latent][latent, out_feat]
			//inp = inp->bypass("aa11\n");
			logits = *inp->dot(wod, { {2},{0} }) + *wod_b;
			/*logits = inp->dot(wod, { {2},{0} });
			logits = logits->bypass("bbb\n");
			logits = *logits + *wod_b;
			logits = logits->bypass("cc33\n");*/
		}
		if(yDiscrete) {
			if(out_feat > 1) logits = logits->reshape({ -1, logits->fshape[1], out_feat, ydisc });
			pred = logits->argmax(-1);//[batch, out_seq, (out_feat)], 잠재코드를 소스 또는 타겟 아이디로 변환
			//pred = pred->bypass("pppppppppppppppppp");
			if(out_feat == 1) {//pred - [batch, out_seq]
				pred = pred->reshape({ pred->fshape[0], pred->fshape[1], 1 });//[batch, out_seq, 1]
			}//else [batch, out_seq, out_feat]
		} else pred = logits;//[batch, out_seq, out_feat]
		if(prtbuild) { 
			printf("out gate logit[batch, out_seq, out_feat] "); logits->shape();
			printf("out gate pred[batch, out_seq, out_feat] "); pred->shape(); 
		}
		tcrAnt->endscope();

		return pred;
	}
	Flux *labelSmoothing(Flux *inp, intt vocab_sz, floatt epsilon)
	{
		auto smoothed = *((1.0 - epsilon) * *inp) + (epsilon / vocab_sz);
		return smoothed;
	}
	Flux *calcLoss(Flux *logit, Flux *label)
	{
		//if(((Trace *)tcrAnt)->nbyout >= 0) {
		//	label = label->slice({ {}, {label->fshape[1] - ((Trace *)tcrAnt)->nbyout, -1} });
		//}
		if(yDiscrete) {//logit - [batch, out_seq, (out_feat), vocab_sz], label - [batch, out_seq, (out_feat|1)]
			Flux *s_label = nullx;
			if(label->fdim < 2 || label->fdim > 3) throwFault(-1, "label dim error\n");
			if(label->fdim == 2) s_label = label;
			else if(label->fshape[label->fdim -1] == 1) s_label = label->squeeze(2);//끝에 1인 차원을 없앤다.
			else if(label->fshape[label->fdim - 1] != logit->fshape[logit->fdim - 2]) {//라벨의 피쳐가 차원이 1이 아닌경우
				//logit - [batch, out_seq, out_feat, vocab_sz], label - [batch, out_seq, out_feat]
				throwFault(-1, "logit: and label dimension inconstant: %d %d\n", 
					logit->fshape[logit->fdim - 2], label->fshape[label->fdim - 1]);
			} else s_label = label;//[batch, out_seq, out_feat]
			//s_label = s_label->bypass("lllllllllllllllllllllllll");
			auto label_ohe = s_label->one_hot(yDiscrete);//[batch, out_seq, (out_feat), vocab_sz]
			//label_ohe = self.label_smoothing(label_ohe, logit.shape[2]);
			if(prtbuild) { printf("calc loss logit & label[batch, out_seq, (out_feat), vocab_sz]\n"); logit->shape(); label_ohe->shape(); }
			logit = logit->softmaxCrossEntropy(label_ohe);
			if(((Trace *)tcrAnt)->batchloss) batchLoss = logit->mean(1);//배치단위 크로스엔트로피 오차 평균 계산
			else batchLoss = nullx;
			return logit->mean();//전체 크로스엔트로피 오차 평균.
		} else {
			if(prtbuild) { printf("calc loss[batch, out_seq, out_feat]\n"); logit->shape();}
			if(((Trace *)tcrAnt)->batchloss) {
				logit = logit->meanSquareError(label, 0);//배치단위 차이제곱 계산
				batchLoss = logit->mean(1);//배치단위 차이제곱 평균 계산
				return logit->mean();//전체 차이제곱 평균 계산
			} else {
				batchLoss = nullx;
				return logit->meanSquareError(label);//logit - [batch, outsz, out_feat]
			}
		}
	}
	Flux *calcAccuracy(Flux *pred, Flux *target, intt discrete_out = -1)
	{
		if(discrete_out < 0) discrete_out = yDiscrete;

		if(discrete_out) {//pred, traget - [batch, out_seq, 1], 타겟 엘레먼트 중에서 zero패드가 아닌 것의 갯수 촣합 분에
			auto target_not_pad = target->not_equal(0.0);//타겟 엘메먼트와 값이 같은 예측 엘레먼트의 갯수 총합 비율 계산
			auto acc = (*(*pred->equal(target) * *target_not_pad) / *target_not_pad->sum())->sum();
			return acc;//1.0이면 정확도 100%
		} else {//pred - [batch, out_seq, vocab_sz]
			return target->squaredDifference(pred)->mean()->sqrt();//차이 제곱 평균 루트, 0.0이면 정확도 100%
		}
	}
	void resizeFhs(void)
	{
		if(entrygate->fshape[0] == lfhstate->ptrFlux->fshape[0]) return;

		for(FluxChain *fch = lfhstate; fch; fch = fch->ptrRight) {
			fch->ptrFlux->resizing2(entrygate, "resize_fhs");
			fch->ptrFlux->fill(0.0);
		}
	}
	Flux *dualEncoderLayer(Flux *pcode, Flux *inp, intt seqsz, intt outsz, intt n_bind, intt n_part,
		intt n_derive, intt kernsz, intt latsz, intt featsz, intt sz_head, intt ni_part, intt psz, 
		intt cross_out, const bytet *name = "dual_encoder_layer")
	{
		Trace *trc = TRACER(tcrAnt);
		intt save_latsz = latsz;//derivefeat이면 밑에서 derive값과 히든 값이 바뀌므로 원 값 보관
		//inp - [batch, party, derive, kernel, featsz]
		trc->namescope(name);
		if(trc->ehook) inp = expand_hook(inp, kernsz, latsz, featsz, n_part, n_derive);
		Flux *hidden;
		if(sz_head) {
			intt nhead = n_derive / sz_head, i = 0;
			auto depart = inp->split(nhead, 2);//[batch, party, derive(nhead*sz_head), kernel, featsz]
			vector<Flux *> hids;
			for(auto iter : *depart) {//[batch, party, sz_head, kernel, featsz]
				auto a = kerneling(iter, 1, trc->derivefeat, kernsz, latsz, featsz, n_part * sz_head, n_part, sz_head, i++);
				hids.push_back(a);//if derivefeat: a - [batch, kernel, party, sz_head, latent] or 
			}					// else [batch, kernel, party, latent, sz_head]
			if(trc->derivefeat) hidden = concat(&hids, 3);//[batch, kernel, party, derive(nhead*sz_head), latent]
			else hidden = concat(&hids, 4);//[batch, kernel, party, latent, derive(nhead*sz_head)]
		} else hidden = kerneling(inp, 1, trc->derivefeat, kernsz, latsz, featsz, n_bind, n_part, n_derive, -1);
		if(trc->derivefeat == 1) {//hidden - [batch, kernel, party, derive, latent]
			intt sz = n_derive;//바로밑에서 derive를 축소하여 cross_out으로 축소하는 개념으로 코드가되있고
			n_derive = latsz;//이 케이스는 latent를 축소하는 경우이므로 서로 바꿔 이하 latent와 derive
			latsz = sz;//를 서로 바뀐 값으로 이해한다.(바로 밑에서 latent가 맨 하위 차원으로 전치된다.)
			if(prtbuild) { printf("dual enocder layer derive feat:%d lattent: %d inversed [batch, party, derive, kernel, featsz]\n", latsz, n_derive); inp->shape(); }
		} else {//TRUE RIGHT. hidden - [batch, kernel, party, latent, derive]
			if(prtbuild) { printf("dual enocder layer[batch, party, derive, kernel, featsz]\n"); inp->shape(); }
		}
		if(trc->inplus) {//스트라이드를 커널사이즈에 중첩되게 할경우 이옵션으로해햐한다. expand후 역으로
			ni_part = 0;//입력 파트와 타겟파트를 분리하는 것이 불가능 하므로
			//입력(없으면 에러)을 포함하는 바이게이트 전채를 목표값으로 학습되도록 0로 설정하여 오토인코딩 
			//학습 되도록 한다. 입력값이 없이 이전코드(pcode)로 하는 경우는 어차피 0이므로 의미없다. 
			//바이게이트에 [|input] + <go token> + target을 주입하고
			//목표값으로 [|input] + target + <end token>을 주입한다.
			//만약 pre-train할거면 바이게이트와 동일하게 목표값을 주입한다.
		}
		Flux *outtens = deriveReduce(hidden, trc->rderivet, kernsz, n_part, latsz, n_derive, ni_part, trc->attsign, cross_out);
		if(trc->derivefeat == 2) outtens = outtens->expand_dims(3);//[batch, party, kernel, 1, latent]
		else outtens = outtens->transpose({ 0, 2, 1, 4, 3 });//[batch, party, kernel, cross_out, latent]
		if(prtbuild) { printf("dual enocder tstep derive reduce[batch, party, kernel, cross_out, latent]\n"); outtens->shape(); }
		if(n_part != outtens->fshape[1] || latsz != outtens->fshape[4] || cross_out != outtens->fshape[3]) exit(1);//체크용 코드
		vector<Flux *> _by_cat, *by_cat, t_cat;
		Flux *kfinal, *kstep;
		Flux *zpad = flux(tcrAnt, { outtens->fshape[0], outtens->fshape[3], outtens->fshape[4] }, lat_tAnt, constant);
		zpad->adjust(inp);//[1,1,32]
		zpad->fill(0.0);
		zpad = zpad->reshape({ -1, 1, 1, cross_out, latsz });//[batch, p, k, cross_out, latent][1,1,1,1,32]
		if(psz) {//하위에서 압축되어 주입되는 코드가있으면 적재
			if(pcode->fshape[2] != latsz) {//이전 문맥코드 차원이 맞지않으면 차원을 맟춘다.
				//(latsz는 위에서 inverse되어 derive size임)
				pcode = pcode->layer_dense(latsz, actfAnt, Initializer::xavier, "pc_inverse");
			}
			pcode = pcode->reshape({ -1, 1, 1, psz, latsz });//[batch, p, k, psz, latent][1,1,1,8,32]
		}//outtens - [batch, party, kernel, cross_out, latent]
		if(ni_part) {//입력파트 스트라이드가 있으면 그 갯수만큼 각 스트라이드 커널의 마지막 k 번째을 발췌하여 적재
			kfinal = outtens->slice({ { }, { 0, ni_part }, { kernsz - 1 } });//예)[1,4,1,1,32]
			by_cat = kfinal->split(ni_part, 1);//ni_part * [batch, p, k, cross_out, latent] 4 * [1,1,1,1,32]
		} else by_cat = &_by_cat;
		for(intt i = ni_part; i < n_part; i++) {//타겟 스트라이드 갯수 * 커널 사이즈 갯수별로 한개 코드출력
			for(intt j = 0; j < kernsz; j++) {
				if(psz) t_cat.push_back(pcode);//[batch, p, k, psz, latent][1,1,1,8,32]
				if(ni_part || i > ni_part) t_cat.insert(t_cat.end(), by_cat->begin(), by_cat->end());//4 * [1,1,1,1,32]
				kstep = outtens->slice({ { }, { i }, { j } });//[batch, p, k, cross_out, latent][1,1,1,1,32]
				t_cat.push_back(kstep);//현 스트라이드 현 커널스텝의 히든 적재
				for(intt k = i + 1; k < n_part; k++) t_cat.push_back(zpad);//이후 스트라이드의 히든은 제로 패딩[1,1,1,1,32]
			}
			kfinal = outtens->slice({ { }, { i }, { kernsz - 1 } });//[batch, p, k, cross_out, latent][1,1,1,1,32]
			by_cat->push_back(kfinal);//다음 스트라이드는 이번 스트라이드의 마지막 커널스텝 양방향(완전)을 참조하게 한다.
		}
		intt nt_part = n_part - ni_part;//타겟파트 스트라이드 갯수
		//nt_part * kernel * (pcode + party) * [batch, p, k, cross_out, latent]4 * 8 * (8 + 8) * [1,1,1,1,32]
		stepcode = concat(&t_cat, 3);//[batch, p, k, nt_part * kernel * (psz + party), cross_out, latent][1,1,1,4*8*(8+8),1,32]
		intt zsz = 1, sz_seq = (psz + n_part) * cross_out;//(psz + party) * cross_out
		if(sz_seq < 8 && trc->zstep == 0) {//reduce할 시퀀스 길이가 7이하이면 파생조합 갯수가
			trc->zstep = 3;//적어 망호출 축소 결과가 좋지못하므로(굳이 망호출할거면 이값을 -1로
		}	//하고 derivefeat를 0로수행, 챗봇길이16으로 입증됨) 망호출에 의한 축소하지 말고 4번케이스로 축소시킨다.
		if(trc->zstep == 3) {//60.스텝별 가중치를 따로하여 스텝 코드 축소
			stepcode = stepcode->reshape({ -1, nt_part * kernsz, sz_seq, latsz });
			stepcode = stepcode->transpose({ 0, 1, 3, 2 });//[batch, nt_part * kernel, latent, sz_seq]

			Flux *wdstep = flux(tcrAnt, { nt_part * kernsz, sz_seq, 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep");
			Flux *wdstep_b = flux(tcrAnt, { nt_part * kernsz, 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep_b");
			//[batch, nt_part * kernel, latent, 1] = [batch, nt_part * kernel, latent, sz_seq] * [nt_part * kernsz, sz_seq, 1] + [party, sizing]
			stepcode = stridedot(stepcode, wdstep, wdstep_b);
			//stepcode = stepcode->reshape({ -1, nt_part, kernsz, 1, latsz });//[batch, nt_part, kernel, 1, latent][1,4,8,1,32]
			stepcode = stepcode->actf(actfAnt);//그래도 nan발생되는 경우있어 불필요, 아마도 나중에 메모리 용량이 되면 derive reduce를 어텐션으로 해야 할듯, 곱셉이후에는 기울기 폭발 방지 목적으로 활성함수를 넣는다.
			if(prtbuild) printf("dual enocder tstep reduce 4\n");
		} else if(trc->zstep == 2) {//60.타겟 파트별 가중치를 따로하여 스텝 코드 축소
			stepcode = stepcode->reshape({ -1, nt_part, kernsz, sz_seq, latsz });
			stepcode = stepcode->transpose({ 0, 2, 1, 4, 3 });//[batch, kernel, nt_part, latent, sz_seq]

			Flux *wdstep = flux(tcrAnt, { nt_part, sz_seq, 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep");
			Flux *wdstep_b = flux(tcrAnt, { nt_part, 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep_b");
			//[batch, kernel, nt_part, latent, 1] = [batch, kernel, nt_part, latent, sz_seq] * [nt_part, sz_seq, 1] + [nt_part, sizing]
			stepcode = dual_stridedot(stepcode, wdstep, wdstep_b);
			stepcode = stepcode->transpose({ 0, 2, 1, 4, 3 });//[batch, nt_part, kernel, 1, latent][1,4,8,1,32]
			stepcode = stepcode->actf(actfAnt);//그래도 nan발생되는 경우있어 불필요, 곱셉이후에는 기울기 폭발 방지 목적으로 활성함수를 넣는다.
			if(prtbuild) printf("dual enocder tstep reduce 3\n");
		} else if(trc->zstep == 1) {//모든 스텝이 가중치를 공유하여 스텝 코드 축소
			stepcode = stepcode->reshape({ -1, nt_part, kernsz, sz_seq, latsz });
			stepcode = stepcode->transpose({ 0, 1, 2, 4, 3 });//[batch, nt_part, kernel, latent, sz_seq]

			Flux *wdstep = flux(tcrAnt, { sz_seq, 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep");
			Flux *wdstep_b = flux(tcrAnt, { 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep_b");
			//[batch, nt_part, kernel, latent, 1] = [batch, nt_part, kernel, latent, sz_seq] * [sz_seq, 1] + [1]
			stepcode = *stepcode->dot(wdstep, { {4}, {0} }) + *wdstep_b;
			//stepcode = stepcode->reshape({ -1, nt_part, kernsz, 1, latsz });//[batch, nt_part, kernel, 1, latent][1,4,8,1,32]
			stepcode = stepcode->actf(actfAnt);//그래도 nan발생되는 경우있어 불필요, 곱셉이후에는 기울기 폭발 방지 목적으로 활성함수를 넣는다.
			if(prtbuild) printf("dual enocder tstep reduce 2\n");
		} else {//trc->zstep == 0, //TRUE RIGHT.망사용 압축
			stepcode = stepcode->reshape({ -1, sz_seq, latsz });//[batch * nt_part * kernel, sz_seq, latent][1*4*8,(8+8)*1,32]
			if(stepcode->fshape[1] > 3) {//각 타입의 압축할 시퀀스 길이가 4이상이면 망으로 압축
				//dualEncoderLayer가 리지주얼로 반복되므로 여기서 재설정되는 모든 trc변수는 복원시킨다.
				floatt low_bound = trc->lowbound;
				intt convolv = trc->convolving, stride_jump = trc->strideJump, kern_size = trc->kernelSize;
				sytet dot_type = trc->dotType, pos_kern = trc->poskern, r_drive = trc->rderivet;
				sytet enc_kern = trc->enckern;
				//스트라이드 간에는 연관 조합이 없으므로 t step별 t시간까지의 각 스트라이드의 마지막 t스뎁으로
				//generic에서 완전 조합 압축코드 생성, 스트라이드가 8개 이상이면 zcode를 2개이상 출력할지 아님
				trc->lowbound = 0.125;//lower bound, 압축 하한선을 8 커널사이즈 당 1 로 설정, 이값을 0.25로
				//하여 8 커널사이즈 당 2가 되어 컨볼빙으로 출력사이즈를 1로되게 해도 출력사이즈가 최소 2이하가 될수없다.
				trc->convolving = 8;//컨볼빙, 입력이 이 사이즈까지 출력압축코드를 1사이즈로 출력, 입력길이가 이 
				//사이즈를 넘으면 8커널단위로 1개길이씩 증가, 예)입력길이 8->1, 16->2, 설정값이 16이면 
				//입력길이 8->1, 16->1, 32->2, 설정값이 32이면 입력길이 8->1, 16->1, 32->1, 64->2 
				if(trc->derivefeat) trc->dotType = STRIDE_DOT;
				else trc->dotType = ORTHO_DOT;//히든(차원)이 컴볼루션(CNN)의 필터(갯수)에 대응됨.
				if(trc->positional > 0) {
					if(trc->poskern == 4) trc->poskern = 2;//입력 시퀀스별로
					//wxh가중치를 르게 하여 위치구분 안하고 대신 포지셔널하는 옵션이면 아래 망 호출에서는 
					//discrete실행이 아니므로 포지셔널이 적용되지 않으므로 wxh가중치로 위치구분하는 옵션실행한다.
				
				} else if(trc->positional < 0) trc->poskern = 6;
				if(trc->rderivet == 0) trc->rderivet = 1;//위 설명과 같이 현재는 아래 망에서 포지셔닝하지 않으므로 aetpr)에서 party가
				//여러개이고 어텐션 곱이 수행될경우에 party간 공유하여 어텐션 곱이 수행되지 않도록 0가 아닌 임의값 설정.
				trc->strideJump = -1;//초기 입력만 설정된 스트라이드로 하고 아래망에서는 기본 스트라이드(width간격)으로 되도록 설정.
				if(trc->szkernDualNest) trc->kernelSize = trc->szkernDualNest;
				if(trc->enckern != 100) trc->enckern = 100;//enckern reset, 듀얼인코더에서 호출되는 코드압축 인코딩은
				if(prtbuild) {								//enckern이 수행되지 않게 한다. 
					printf("dual enocder tstep reduce 0 conv: 8 lbound: 0.125 nt_part: %d kernel: %d psz: %d party: %d reducing: %d [batch * nt_part * kernel, (psz + party) * reducing, latent]\n",
						nt_part, kernsz, psz, n_part, cross_out); stepcode->shape();
				}
				trc->nbyout = -1;//dual incoder off//stepcode - [batch * nt_part * kernel, sz_seq, latent]
				Generic *seqout = generic(stepcode, nullx, save_latsz, 0, 0, 0, -1, -1, "dual_seqout");
				stepcode = seqout->zcodec;//[batch * nt_part * kernel, zcode, latent][1*4*8,z,32]
				latsz = stepcode->fshape[2];//derivefeat인 경우 시퀀스가 8이 안되면 파생조합 적어지므로 재설정.
				trc->lowbound = low_bound; trc->convolving = convolv;
				trc->dotType = dot_type; trc->poskern = pos_kern;
				trc->rderivet = r_drive; trc->strideJump = stride_jump;
				trc->kernelSize = kern_size;
				trc->nbyout = outsz;//dual incoder on, 후에 class loss에서 또 사용하므로 복원
				trc->enckern = enc_kern;
			}
			zsz = stepcode->fshape[1];
		}
		if(prtbuild) { printf("dual enocder layer nt_part: %d kernel: %d zsz: %d [batch * nt_part * kernel, zcode, latent] ", nt_part, kernsz, zsz); stepcode->shape(); }
		intt nt_seq = nt_part * kernsz * zsz;//stepcode - [batch * nt_part * kernel, zcode, latent][1*4*8,z,32]
		Flux *t_code = stepcode->reshape({ -1, nt_seq, latsz });//[batch, nt_seq, latent][1,4*8*z,32]
		if(prtbuild) { printf("dual enocder layer[batch, nt_part * kernsz * zsz, latent] "); t_code->shape(); }
		if(trc->dualatten) {
			stepcode = stepcode->reshape({ -1, nt_part, kernsz, zsz, latsz });//[batch, nt_part, kernel, zcode, latent][1,4,8,z,32]
			//매 스트라이드의 양방향(완전 조합)인 마지막 커널스텝들만 모아서 키를 구성한다.
			stepcode = stepcode->slice({ { }, { }, { kernsz - 1 } });//[batch, nt_part, k, zcode, latent][1,4,1,z,32]
			stepcode = stepcode->reshape({ -1, nt_part * zsz, latsz });//[batch, nt_part*k*zcode, latent][1,4*1*z,32]
			//순차(일방향)스텝을 쿼리로 양방향(완전조합)을 키로 어텐션 수행하여 각 시퀀스에서 전체를 고려헌 출력을 기대한다.
			t_code = attention_layer2(t_code, stepcode, latsz);//[batch, nt_seq, latent][1,4*8*z,32]
		}
		t_code = stride_fit(t_code, trc->inplus ? seqsz : outsz);//inplus이면 출력
		//사이즈는 바이게이트(inp) 전제 사이즈인 seqsz가 된다. outsz는 목표값만의 사이즈이다.
		//위에서 스트라이드를 커널사이즈보다 적게했으면 중첩되어 stepcode의 파트갯수가 늘어나므로
		//출력 사이즈에 맞춘다.
		if(prtbuild) { printf("dual enocder layer out[batch, out seq, latent] "); t_code->shape(); }
		trc->endscope();

		return t_code;
	}
	Flux *stride_fit(Flux *out, intt nseq)
	{
		Trace *trc = TRACER(tcrAnt);
		intt feat = out->fshape[2];

		if(nseq != out->fshape[1]) {
			if(prtbuild) { printf("stride fit:%d -> %d[batch, out seq, latent]\n", out->fshape[1], nseq); out->shape(); }
			if(out->fshape[1] > nseq && out->fshape[1] % nseq == 0) {//나중에 conv1d구현하여
				//정합되지 않아도 시퀀스 맞춤할 수 있게 한다.
				//feat = (out->fshape[1] / nseq) * feat;
				out = out->reshape({ -1, nseq, (out->fshape[1] / nseq) * feat });//[batch, nseq, feat]
				out = out->layer_dense(feat, actfAnt, Initializer::xavier, "ffd_seq");
			} else {
				out = out->transpose({ 0, 2, 1 });//[batch, feat, seq] <- [batch, seq, feat]
				out = out->layer_dense(nseq, actfAnt, Initializer::xavier, "ffd_seq");//[batch, feat, nseq]
				out = out->transpose({ 0, 2, 1 });//[batch, nseq, feat]
			}
			if(prtbuild) out->shape();
		}
		return out;
	}
	Flux *feed_forward(Flux *out)
	{
		Trace *trc = TRACER(tcrAnt);
		intt feat = out->fshape[2];

		if(trc->dualdff > 0) {
			out = out->layer_dense(trc->dualdff, actfAnt, Initializer::xavier, "ffd_dense");
			out = out->layer_dense(feat, actfAnt, Initializer::xavier, "ffd_dense2");
		} else if(trc->dualdff < 0) {
			auto r = attention_layer(out, feat, "ffd_attention");
			out = *out + *r;
			out = attention_layer(out, feat, "ffd_attention2");
			if(trc->dualdff < -1) {
				out = out->layer_dense(trc->dualdff * -1, actfAnt, Initializer::xavier, "ffd_dense");
				out = out->layer_dense(feat, actfAnt, Initializer::xavier, "ffd_dense2");
			}
		}
		return out;
	}
	//eternal code(icode)추가방안
	//1.large demension 1개 길이 icode를 준비한다.
	//2.pcode를 길이가 7개가 되게 한다.(8길이씩 압축된다면 pcode최초입력 사이즈 448 -> 56 -> 7로서 두번 압축)
	//3.pcode의 차수를 icode의 차수와 같게 늘려 pcode2 icode의 1개 길이를 pcode의 길이 7개로 늘려 icode2.
	//4.pocde2를 query, icode2를 key, icode2의 7길이와 출력 디멘젼(pcode의 디멘젼)으로 하는 플럭스를 생성하여 value attension곲한다.
	//5.또는 pcode를 query, 1개길이 icode를 value로하여 query * value.T를 생성, value의 길이가 1개이므로 소프트맥스없이 
	//	1개길이에 출력 디멘젼(pcode의 디멘젼)으로 하는 플럭스를 생성하여 [query * value.T]에 곱하여 icode가 query된 pcode생성
	//6.icode와 pocde2를 보관하고 5)에서 생성된 pcode로 듀얼인코더를 수행하여 역전파된후 역전파 반영된 icode와 pocde2를 
	//	다음 스텝 실행에서 cat하여 8길이로 만들어 망 압축하여 1개 길이 icode생성하고 이전 icode와 리지주얼
	//7. 6)에서 생성된 icode로 2)번부터 수행.
	Flux *eternalCode(Flux *pcode)
	{
		Trace *trc = TRACER(tcrAnt);
		intt fsz = trc->icodefz;
		intt npseq = pcode->fshape[1];

		iCode = flux(tcrAnt, { 1, fsz }, lat_tAnt, trainable, Initializer::xavier, "i_code");
		ppCode = flux(tcrAnt, { -1, npseq, fsz }, lat_tAnt, persistant);
		//auto rcode = ipCode->transpose({ 1, 0, 2 });//[1, batch, fsz]
		//rcode = rcode->reshape({ 1, rcode->fshape[1] * fsz });//[1, batch * fsz]
		//iCode = rcode->layer_dense(fsz, actfAnt, Initializer::xavier, "ec_dense");//[1, fsz]
		//이하는 영속코드(iCode) 학습 및 대화 영속성(history)는 유지하는 추론 코드
		ppCode->adjust(pcode);//첫번째 실행에서 ppCode의 배치를 pcode와 일치되게 확장.
		auto ei_code = flux(tcrAnt, { -1, 1, fsz }, lat_tAnt, constant);
		ei_code->fill(1.0, pcode);
		ei_code = ei_code->mul(iCode);//[batch, 1, fsz] = [1, fsz], icode 배치 확장
		ei_code = concat({ ei_code, ppCode }, 1);//[batch, 1 + npseq, fsz]
		//망 압축 begin, iCode는 역전파에 의해 반영된다.
		floatt low_bound = trc->lowbound;
		intt convolv = trc->convolving, stride_jump = trc->strideJump, dout = trc->nbyout;
		sytet dot_type = trc->dotType, pos_kern = trc->poskern, pos = trc->positional;
		//스트라이드 간에는 연관 조합이 없으므로 t step별 t시간까지의 각 스트라이드의 마지막 t스뎁으로
		//generic에서 완전 조합 압축코드 생성, 스트라이드가 8개 이상이면 zcode를 2개이상 출력할지 아님
		trc->lowbound = 0.125;//lower bound, 압축 하한선을 8 커널사이즈 당 1 로 설정, 이값을 0.25로
		//하여 8 커널사이즈 당 2가 되어 컨볼빙으로 출력사이즈를 1로되게 해도 출력사이즈가 최소 2이하가 될수없다.
		trc->convolving = 8;//컨볼빙, 입력이 이 사이즈까지 출력압축코드를 1사이즈로 출력, 입력길이가 이 
		//사이즈를 넘으면 8커널단위로 1개길이씩 증가, 예)입력길이 8->1, 16->2, 설정값이 16이면 
		//입력길이 8->1, 16->1, 32->2, 설정값이 32이면 입력길이 8->1, 16->1, 32->1, 64->2 
		trc->dotType = ORTHO_DOT;//히든(차원)이 컴볼루션(CNN)의 필터(갯수)에 대응됨.
		trc->poskern = 3;
		trc->positional = 0;
		trc->strideJump = -1;//초기 입력만 설정된 스트라이드로 하고 아래망에서는 기본 스트라이드(width간격)으로 되도록 설정.
		trc->nbyout = -1;//dual incoder off
		Generic *icout = generic(ei_code, nullx, fsz, 0, 0, 0, -1, -1, "icode_update");
		ei_code = icout->zcodec;//[batch, 1, fsz], [영속코드 + 이전이전코드]시퀀스를 망 압축
		trc->lowbound = low_bound; trc->convolving = convolv;
		trc->dotType = dot_type; trc->poskern = pos_kern;
		trc->positional = pos; trc->strideJump = stride_jump;
		trc->nbyout = dout;
		//망 압축 end, ei_code = [batch, 1, fsz]
		ei_code = ei_code->transpose({ 0, 2, 1 });//[batch, fsz, 1]
		ei_code = ei_code->layer_dense(pcode->fshape[2], actfAnt, Initializer::xavier, "ec_dense");//[batch, fsz, feat]
		auto ep_code = pcode->layer_dense(fsz, actfAnt, Initializer::xavier, "ec_dense2");//[batch, npseq, fsz]
		//auto pp_code = pcode->matmul(ei_code, 2);//[batch, npseq, fsz] = [batch, npseq, feat] * [batch, feat, fsz]
		auto pp_code = *ep_code + *ppCode;//[batch, npseq, fsz], residual(역전파는 없음) & 영속성 지원, 영속성을 위해 위 스텝과 현 스텝 중 택일
		pp_code = pp_code->actf(actfAnt);
		ovCode = pp_code->overwrite(ppCode);//[batch, npseq, fsz], ppCode에의해 대화의 영속성 지원된다.
		teternal = ep_code->matmul(ei_code);//[batch, npseq, feat] = [batch, npseq, fsz] * [batch, fsz, feat]
		teternal = teternal->actf(actfAnt);
		//이하는 대화 영속성(history)는 유지하지 않고 eternal code(icode)와 pcode에 의해 추론하는 코드
		auto ei_code2 = flux(tcrAnt, { -1, 1, fsz }, lat_tAnt, constant);
		ei_code2->fill(1.0, pcode);
		ei_code2 = ei_code2->mul(iCode);//[batch, 1, fsz] = [1, fsz], icode 배치 확장
		ei_code2 = ei_code2->transpose({ 0, 2, 1 });//[batch, fsz, 1]
		ei_code2 = ei_code2->layer_dense(pcode->fshape[2], actfAnt, Initializer::xavier, "ec_dense3");//[batch, fsz, feat]
		auto ep_code2 = pcode->layer_dense(fsz, actfAnt, Initializer::xavier, "ec_dense4");//[batch, npseq, fsz]
		reternal = ep_code2->matmul(ei_code2);//[batch, npseq, feat] = [batch, npseq, fsz] * [batch, fsz, feat]
		reternal = reternal->actf(actfAnt);

		return teternal->switchout(reternal);
	}
	//pcode - 현 듀얼인코더에서 완전조합 전개되지 않는 입력 파트 코드, inp - (입력+추론출력), 
	//outsz - 추론출력만의 길이로서 inp의 시퀀스에서 outsz를 뺀 사이즈는 입력파트로서 추론출력과 함께 
	//완전조합 전개되고 이 입력파트 길이를 strideAnt의 배수가 되게하고 최소 kernel보다 크게 주입하여 
	//time전개되는 kernel 경계에 걸리지 않게 한다. inp시퀀스 길이가 outsz와 같으면 inp는 추론출력만 
	//주입되는 것이고 입력은 pcode로만 주입된다, 추론출력 - <s> + 타겟
	//chatbot예제에서 <s> 시작토큰을 데이터에서 제거하고 실행한다. 시작토큰은 0값으로서 
	//auto regression에서 go 토큰으로 사용되기 때문에
	//indiscret는 이산데이터일 경우 의미있고 이때 inp가 입력과 타겟 쌍일 경우 둘을 합한 discrete사이즈
	//이고 입력파트(ni_part)가 없고 타겟만 있을 경우 타겟만의 discrete사이즈이다.
	Flux *dualEncoder(Flux *pcode, Flux *inp, intt outsz, intt indiscret, intt outdiscret, intt embedim, sytet interv = -1, intt kernsz = -1,
		intt latsz = -1, const bytet *name = "anet_dual_encode")
	{
		Trace *trc = TRACER(tcrAnt);
		bytet s[64];
		Flux *stepcode, *out, *expand_mask;
		intt seqsz = inp->fshape[1], featsz = inp->fshape[2], noc = 0;
		intt insz = seqsz - outsz;
		if(insz < 0) throwFault(-1, "out size(%d) is longer than by gate(%d) error \n", outsz, seqsz);
		if(entrygate == nullx) entrygate = inp;
		if(interv < 0) interv = intervAnt;
		if(kernsz < 0) kernsz = kernszAnt;
		if(latsz < -1) latsz *= -1;
		else if(latsz <= 0) latsz = latszAnt;
		//inp == [batch, seqsz, featsz]
		trc->namescope(name);
		if(seqsz <= 4) kernsz = 4;
		intt n_derive, sz_head = 0, width = (kernsz < widthAnt ? kernsz : widthAnt);
		floatt r_contig = (width < 8 && rExContigAnt < 0.7 ? 0.7 : rExContigAnt);
		lat_tAnt = inp->qType;
		if(prtbuild) {
			if(pcode) { printf("dual enocder prev code: "); pcode->shape(); }
			printf("dual enocder by gate "); inp->shape();
		}
		//가급적이면 시퀀스 길이를 커널 사이즈/스트라이드의 배수로 하여 두번째 케이스로 실행한다.
		//시퀀스와 커널이 정합되지 않으면(스트라이드에 따라 아닐수도 있지먄, 예를 들어 시퀀스32, 커널8, 
		//스트라이드6이면 스트라이드 4번째(옵셋 24)에서 멈추어 제로패딩하지 않음) 제로페딩이 설정되므로 학습 효율이 떨어질것.
		if(trc->ebatch_t == 2) {
			Flux *a;
			a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			a->fill(1.0);
			expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - 스트라이드 단위 파생 시퀀스 갯수
			//expand_mask->shape();
			//expand_mask->printo(1,2);
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			inp = expand_batch2(inp, expand_mask, strideAnt, kernsz, zpadAnt, noc);//배치의 모든 시퀀스들의 파생 조합 시퀀스 생성
			//inp->printo(1, 2);
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, strideAnt, r_contig, zpadAnt, 0, 0);//체크용 코드
			//if(a->fshape[0] != inp->fshape[1]) throwFault(-1, "check error\n");
		} else {
			auto *a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			a->fill(1.0);
			expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - 스트라이드 단위 파생 시퀀스 갯수
			//expand_mask->shape();
			//expand_mask->printo();
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			inp = expand_batch(inp, expand_mask, interv, kernsz);//배치의 모든 시퀀스들의 파생 조합 시퀀스 생성
			//inp->printo(1, 2);
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, stride, r_contig, zpadAnt, 0, 0);//체크용 코드
			//if(a->fshape[0] != inp->fshape[1]) throwFault(-1, "check error\n");
		}//inp == [batch, bindbatch(party*derive), kernel, featsz]
		//커널 사이즈와 expand를 이원화하여 확장은 2의 8승으로 하고 커널스텝은 16내지 32로 하여 
		//스트라이드 2개 혹은 4개를 8커널스텝 단위로 의미가 모아지고 다음 스트라이드는 이전의 모아진 의미
		//로부터 시작된다. 상위로 올리는 것은 마지막 커널 스텝만 올린다.
		//residual을 위해 모든 피쳐 사이즈를 일치시킨다.
		if(featsz > latsz) latsz = featsz;//피쳐값중 제일 큰 것을 latsz으로 설정.
		if(pcode) {
			if(trc->eternity) pcode = eternalCode(pcode);
			if(pcode->fshape[pcode->fdim - 1] > latsz) latsz = pcode->fshape[pcode->fdim - 1];
		}
		if(indiscret > 0) {//입력 임베딩
			if(inp->fshape[3] * embedim >= latsz) {//혹시 입력 피쳐가 복수차원일경우 그 각각을 임베딩하여
				latsz = inp->fshape[3] * embedim;//하나로 합친 사이즈가 히든사이즈보다 크면 히든을 그사이즈로 늘린다.
			} else {//하나로 합친사이즈(복수차원이 아니면 1개사이즈)보다 히든이 더크면 임베딩사이즈를 늘려서
				embedim = latsz / inp->fshape[3];//히든 사이즈에 맟춘다.
				latsz = inp->fshape[3] * embedim;
			}
			if(inp->fshape[inp->fdim - 1] == 1) inp = inp->squeeze(inp->fdim - 1);//[batch, bindbatch(party*derive), kernel]
			Flux *expand = nullx;
			if(trc->dualEmbed < 3 && (inp->fshape[1] % n_derive) == 0) {//입력과 출력 구분이 n_derive단위로 맞아떨어야함
				intx nby = inp->fshape[1] / n_derive;
				intx step = seqsz / nby;
				intx in_derive = (insz / step) * n_derive;//바이게이트에서 입력 파트의 파생 시퀀스 갯수
				if(trc->dualEmbed == 0 || trc->dualEmbed == 2) {//입력 파트만 임베딩 혹은 입력과 출력 각각 임베딩
					auto embed = inp->slice({ {}, {0, in_derive} });
					expand = inp->slice({ {}, {in_derive, -1} });//출력파트는 단순 차원 확장 혹은
					if(trc->dualEmbed == 2) expand = embedding(expand, outdiscret, embedim, 1);//출력 파트 따로 임베딩
					inp = embed;
				} else {//출력 파트만 임베딩
					expand = inp->slice({ {}, {0, in_derive} });//입력파트는 단순 차원 확장
					inp = inp->slice({ {}, {in_derive, -1} });
				}
				if(trc->dualEmbed != 2) {
					expand = expand->expand_dims(-1);
					expand = expand->layer_dense(embedim, actfAnt, Initializer::xavier, "inp_embed");
				}
			} 
			inp = embedding(inp, indiscret, embedim, 1);//[batch, bindbatch(party*derive), kernel, featsz(embedim)]
			if(trc->dualEmbed < 3) {
				if(trc->dualEmbed == 0 || trc->dualEmbed == 2) inp = concat({ inp, expand }, 1);//입력 파트 임베딩과 출력파트 단순확장 붙임
				else inp = concat({ expand, inp }, 1);//입력파트 단순확장과 출력파트 임베딩 붙임
			}
			//입력 피쳐가 복수차원이면 각 피쳐를 임베딩후에 피쳐를 한개로 합한다.
			if(inp->fdim > 4) inp = inp->reshape({ inp->fshape[0], inp->fshape[1], inp->fshape[2], -1 });
			if(inp->fshape[inp->fdim - 1] != latsz) {
				printf("illegal embedim\n");
				exit(1);
			}
			featsz = latsz;
		} else if(featsz != latsz) {//inp가 타겟만 있고 추론할때라서 아무것도 없고
			featsz = latsz;//go토큰 하나만 있는 경우라도 일치시킨다.
			inp = inp->layer_dense(featsz, actfAnt, Initializer::xavier, "inp_feat");
		}//inp - [batch, bindbatch(party*derive), kernel, featsz(embedim)]
		intt n_bind = inp->fshape[1];//한개 시퀀스의 총 파생 조합 시붠스 갯수
		intt n_part = n_bind / n_derive;//seqsz / kernsz
		if(trc->positional > 0) {
			trc->positional = 0;//처음 한번만 포지셔널
			inp = inp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
			inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
			auto enpos = flux(trc, 3, &inp->fshape[2], inp->qType, constant);//[party, kernel, featsz]
			enpos->sinpos(n_part * kernsz);//[seq(party, kernel), featsz]
			inp = *inp + *enpos;//sinuosid positional, [batch, derive, party, kernel, featsz] = [batch, derive, party, kernel, featsz] + [positional(party, kernel), featsz]
			inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, party, derive, kernel, featsz]
		} else inp = inp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
		
		//seqsz(입력+타겟)에서 입력파트 길이을 strideAnt의 배수가 되게하고 최소 kernel보다 크게 주입하여
		if(seqsz % n_part || (seqsz - outsz) % (seqsz / n_part)) {//time전개되는 kernel경계에 
			throwFault(-1, "dual kernel size not align\n");//걸리지 않게 한다. ni_part가 0이면
		}									//입력파트는 완전조합 전개되지 않고 pcode로만 참조된다.
		intt psz = (pcode ? pcode->fshape[1] : 0);
		intt cross_out = trc->szDualReduce;//커널 윈도우내 파생 조합 시퀀스 단위별로 축소되는 출력 갯수
		intt ni_part = (seqsz - outsz) / (seqsz / n_part);//입력파트의 스트라이드 갯수
		if(prtbuild) printf("dual enocder bysz: %d tsz: %d p_sz: %d stride: %d kernel: %d derive: %d ni_part: %d party: %d reducing: %d outsz: %d\n",
			seqsz, outsz, psz, strideAnt, kernsz, n_derive, ni_part, n_part, cross_out, outsz);
		//inp - [batch, party, derive, kernel, featsz]
		Flux *in_part = nullx, *tar_part = nullx;
		if(trc->nblockChain > 1 && trc->inplus == 0 && insz) {//residual옵션인데 inplus가 0이면 타겟값
			//만 출력하는 옵션인데, 이에 더하여 바이게이트에 입력값이 있으면 입력값 부분만 발췌하여 밑에서 
			//리지주얼할때 입력값을 합친다. 스트라이드를 커널사이즈에 중첩되게 할경우 이옵션으로 하면않된다.
			//위에서 확장된후에 입력파트만 추출하는 것도 expand후 역으로 입력 파트와 타겟파트를 분리하는 것도
			in_part = inp->slice({ { }, { 0, ni_part} }); //불가능 하므로, 중첩할렴녀 inplus옵션으로 실행한다.
			if(trc->resiFirst) tar_part = inp->slice({ { }, { ni_part, -1 } });
		} else if(trc->resiFirst) tar_part = inp;
		if(trc->nblockChain == 0) trc->nblockChain = 1;
		out = inp;
		for(intt i = 0; i < trc->nblockChain; i++) {
			sprintf(s, "dual_chain_%d", i);
			stepcode = dualEncoderLayer(pcode, out, seqsz, outsz, n_bind, n_part, n_derive, 
				kernsz, latsz, featsz, sz_head, ni_part, psz, cross_out, s);
			if(i + 1 < trc->nblockChain) {// || trc->resiopt) {//마지막을 다시 확장하면 형태가 [batch, seq, feat]
				//가 되질 않으므로 타겟과 연결 학습을 할수없으므로[ ㅁ)과 함께 제거하여 ] 중간만 수행, 
				//나중에 확장후 리지주얼후 모양을 변경하는것 검토.
				if(trc->ebatch_t == 2) stepcode = expand_batch2(stepcode, expand_mask, strideAnt, kernsz, zpadAnt, noc);
				else stepcode = expand_batch(stepcode, expand_mask, interv, kernsz);
				stepcode = stepcode->reshape({ -1, n_part - ni_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
			}
			if(trc->resiFirst == 0) {//입력을 먼저 합한후 리지주얼
				if(i + 1 < trc->nblockChain) {//중간 인코더 블럭이면 위에서 발췌한 입력파트가 있으면 
					if(in_part) stepcode = concat({ in_part, stepcode }, 1);//이를 출력에 합침.
				}// else {//ㅁ.마지막 인코더 블럭이면 바이게이트가 in+target인 경우 이전 마지막 출력에서 타겟파트만
				//	//발췌하여 밑에서 리지주얼이면 타겟파트만 더할수 있게 한다.
				//	if(trc->resiopt && in_part) out = out->slice({ { }, { ni_part, -1} });
				//}
				if(trc->resiopt && i + 1 < trc->nblockChain) {//중간인코더이면 입력+타겟 전체를 이전것과 
					out = *out + *stepcode;//더하는 것이고 마지막 인코더이면 타겟만 더함
					if(trc->dual_lnorm) {
						sprintf(s, "dual_lnorm_%d", i);
						out = out->layer_normal(s);
					}
					out = out->actf(actfAnt);
				} else out = stepcode;
			} else {//리지주얼후 입력 합
				if(i + 1 < trc->nblockChain) {//중간 인코더 블럭이면 위에서 발췌한 입력파트가 있으면
					if(trc->resiopt) stepcode = *tar_part + *stepcode;
					if(in_part) out = concat({ in_part, stepcode }, 1);//이를 출력에 합침.
					else out = stepcode;
					if(trc->resiopt) {
						if(trc->dual_lnorm) {
							sprintf(s, "dual_lnorm_%d", i);
							out = out->layer_normal(s);
						}
						out = out->actf(actfAnt);
						if(in_part) tar_part = out->slice({ { }, { ni_part, -1 } });
						else tar_part = out;
					}
				} else out = stepcode;
			}
			out = feed_forward(out);
		}
		trc->endscope();
		return out;
	}
	//pcode - 현 듀얼인코더에서 완전조합 전개되지 않는 입력 파트 코드, inp - (입력+추론출력), 
	//outsz - 추론출력만의 길이로서 inp의 시퀀스에서 outsz를 뺀 사이즈는 입력파트로서 추론출력과 함께 
	//완전조합 전개되고 이 입력파트 길이를 strideAnt의 배수가 되게하고 최소 kernel보다 크게 주입하여 
	//time전개되는 kernel 경계에 걸리지 않게 한다. inp시퀀스 길이가 outsz와 같으면 inp는 추론출력만 
	//주입되는 것이고 입력은 pcode로만 주입된다, 추론출력 - <s> + 타겟
	//chatbot예제에서 <s> 시작토큰을 데이터에서 제거하고 실행한다. 시작토큰은 0값으로서 
	//auto regression에서 go 토큰으로 사용되기 때문에
	//indiscret는 이산데이터일 경우 의미있고 이때 inp가 입력과 타겟 쌍일 경우 둘을 합한 discrete사이즈
	//이고 입력파트(ni_part)가 없고 타겟만 있을 경우 타겟만의 discrete사이즈이다.
	Flux *chainEncoder(Flux *pcode, Flux *inp, intt outsz, intt indiscret, intt embedim, intt kernsz = -1,
		intt latsz = -1, const bytet *name = "anet_chain_encode")
	{
		Trace *trc = TRACER(tcrAnt);
		if(entrygate == nullx) entrygate = inp;
		if(kernsz < 0) kernsz = kernszAnt;
		if(latsz < -1) latsz *= -1;
		else if(latsz <= 0) latsz = latszAnt;
		lat_tAnt = inp->qType;

		trc->namescope(name);
		intt seqsz = inp->fshape[1], featsz = inp->fshape[2];
		//inp == [batch, seqsz, featsz], residual을 위해 모든 피쳐 사이즈를 일치시킨다.
		if(featsz > latsz) latsz = featsz;//피쳐값중 제일 큰 것을 latsz으로 설정.
		if(pcode && pcode->fshape[pcode->fdim - 1] > latsz) latsz = pcode->fshape[pcode->fdim - 1];
		if(indiscret > 0) {//입력 임베딩, 입력 피쳐가 복수차원일 경우 세번째 차원의 각 원소가 임베딩된다.
			if(inp->fshape[2] * embedim >= latsz) latsz = inp->fshape[2] * embedim;
			else {
				embedim = latsz / inp->fshape[2];
				latsz = inp->fshape[2] * embedim;
			}
			if(inp->fshape[inp->fdim - 1] == 1) inp = inp->squeeze(inp->fdim - 1);
			inp = embedding(inp, indiscret, embedim, 1);//[batch, seqsz, featsz(embedim)]
			//입력 피쳐가 복수차원이면 각 피쳐를 임베딩후에 피쳐를 한개로 합한다.
			if(inp->fdim > 3) inp = inp->reshape({ inp->fshape[0], inp->fshape[1], -1 });
			if(inp->fshape[inp->fdim - 1] != latsz) {
				printf("illegal embedim\n");
				exit(1);
			}
			featsz = latsz;
		} else if(featsz != latsz) {//inp가 타겟만 있고 추론할때라서 아무것도 없고
			featsz = latsz;//go토큰 하나만 있는 경우라도 일치시킨다.
			inp = inp->layer_dense(latsz, actfAnt, Initializer::xavier, "inp_feat");
		}
		Flux *stepcode = inp;
		bytet s[64];
		if(pcode) {
			if(pcode->fshape[pcode->fdim - 1] != latsz) {
				pcode = pcode->layer_dense(latsz, actfAnt, Initializer::xavier, "pcode_feat");
			}
			stepcode = concat({ pcode, stepcode }, 1);
		}
		if(trc->nblockChain == 0) trc->nblockChain = 1;
		for(intt i = 0; i < trc->nblockChain; i++) {
			sprintf(s, "ceb_chain_%d", i);
			if(trc->dualChain == 1) stepcode = nestChain(stepcode, latsz, -1, s);
			else stepcode = blockChain(stepcode, latsz, -1, s);
		}
		//바이게이트에 [|input] + <go token> + target을 주입하고~
		if(trc->inplus) {//입력(없으면 에러)을 포함하는 바이게이트 전채를 목표값으로 하는 경우인데
			if(seqsz != stepcode->fshape[1]) {//위에서 이전코드가 합해졋으면 바이게이트 부분만 추출한다.
				stepcode = stepcode->slice({ { }, { stepcode->fshape[1] - seqsz, -1} });
			}//else 이전코드가 없는 경우, stepcode는 바이게이트와 동일 사이즈이므로 그대로
			//~목표값으로 input + target + <end token>을 주입하여 feed해야 한다.
			//만약 pre-train할거면 바이게이트와 동일하게 목표값을 주입한다.
		} else if(seqsz != outsz) {//outsz는 목표값 사이즈, 바이게이트에 입력이 포함되있던 없던 목표값만 추출
			stepcode = stepcode->slice({ { }, { stepcode->fshape[1] - outsz, -1} });
			//~목표값으로 target + <end token>을 주입하여 feed해야 한다.
		}
		if(prtbuild) { printf("chain enocder out[batch, out seq, latent]\n"); stepcode->shape(); }
		trc->endscope();

		return stepcode;
	}
	Flux *nestChain(Flux *inp, intt latsz, intt kernsz = -1, const bytet *name = "nest_chain")
	{
		Trace *trc = TRACER(tcrAnt);
		ubytet dat_t = inp->qType;
		intt noc = 0;
		if(kernsz < 0) kernsz = kernszAnt;

		trc->namescope(name);
		intt seqsz = inp->fshape[1], featsz = inp->fshape[2], n_derive, sz_head = 0;
		Flux *dexp, *residual = nullx;
		if(seqsz <= 4) kernsz = 4;//시퀀스가 4보다 크면 커널은 주어진 커널 사이즈로 되고 시퀀스가
									//커널에 정합되지 않으면 모자르는 것은 제로패딩 확장된다.
		intt nfinal = (seqsz / kernsz >= 4 ? seqsz / kernsz : 0), resiopt = trc->resiopt;
		if(nfinal && (resiopt == 1 || resiopt == 3)) {//각 파트 마지막 커널스텝에 해당하는 입력 추출, input or all residual
			residual = inp->reshape({ -1, nfinal, kernsz, featsz });//[batch, party, kernsz, featsz]
			residual = residual->slice({ { }, { }, { kernsz - 1 } });//[batch, party, kfinal, featsz]
			residual = residual->reshape({ -1, nfinal, featsz });//[batch, party, featsz]
		}
		intt width = (kernsz < widthAnt ? kernsz : widthAnt);
		//가급적이면 시퀀스 길이를 커널 사이즈/스트라이드의 배수로 하여 두번째 케이스로 실행한다. 시퀀스와 커널이 정합되지 
		//않으면(스트라이드에 따라 아닐수도 있지먄, 예를 들어 시퀀스32, 커널8, 스트라이드6이면 스트라이드 4번째(옵셋 24)에서 
		//멈추어 제로패딩하지 않음) 제로페딩이 설정되므로 학습 효율이 떨어질것.
		floatt r_contig = (width < 8 && rExContigAnt < 0.7 ? 0.7 : rExContigAnt);//폭이작고 rate가 적으면 조합갯수가 0가 되므로 이때는 비율을 높힌다.
		Flux *a = flux(tcrAnt, { 1, width, featsz }, dat_t, constant);
		a->fill(1.0);
		auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
		n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - 스트라이드 단위 파생 시퀀스 갯수
		//expand_mask->shape();
		//expand_mask->printo(1,2);
		//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
		dexp = expand_batch2(inp, expand_mask, strideAnt, kernsz, zpadAnt, noc);//배치의 모든 시퀀스들의 파생 조합 시퀀스 생성

		intt n_bind = dexp->fshape[1];//한개 시퀀스의 총 파생 조합 시붠스 갯수
		intt n_part = n_bind / n_derive;//transform stride단위가 시퀀스 길이에 정수배로 맞아 떨어지게 해야함, 아니면 transform에서 zero padding설정해야함.
		if(prtbuild) printf("nest chain seq sz: %d feat sz: %d stride: %d kernel: %d derive: %d party: %d\n", seqsz, featsz, strideAnt, kernsz, n_derive, n_part);
		if(tcrAnt->dbgStep == 1) dexp = dexp->bypass("111\n");
		if(trc->positional > 0) {
			if(trc->positional == 1) trc->positional = 0;//처음 한번만 포지셔널, 음수이면 이하 리지주얼 블럭 계속 포지셔널
			dexp = dexp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
			dexp = dexp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
			auto enpos = flux(trc, 3, &dexp->fshape[2], dexp->qType, constant);//[party, kernel, featsz]
			enpos->sinpos(n_part * kernsz);//[seq(party, kernel), featsz]
			dexp = *dexp + *enpos;//sinuosid positional, [batch, derive, party, kernel, featsz] = [batch, derive, party, kernel, featsz] + [positional(party, kernel), featsz]
			dexp = dexp->transpose({ 0, 2, 1, 3, 4 });//[batch, party, derive, kernel, featsz]
		} else {
			dexp = dexp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
		}
		Flux *hidden, *stepcode;
		if(sz_head) {
			intt nhead = n_derive / sz_head, i = 0;
			auto depart = dexp->split(nhead, 2);//[batch, party, derive(nhead*sz_head), kernel, featsz]
			vector<Flux *> hids;
			for(auto iter : *depart) {//[batch, party, sz_head, kernel, featsz]
				auto a = kerneling(iter, 1, 0, kernsz, latsz, featsz, n_part * sz_head, n_part, sz_head, i++);
				hids.push_back(a);//[batch, kernel, party, latent, sz_head]
			}
			hidden = concat(&hids, 4);//[batch, kernel, party, latent, derive(nhead*sz_head)]
		} else hidden = kerneling(dexp, 1, 0, kernsz, latsz, featsz, n_bind, n_part, n_derive, -1);
		//hidden - [batch, kernel, party, latent, derive]
		if(prtbuild) { printf("nest chain[batch, kernel, party, latent, derive]\n"); hidden->shape(); }
		auto reduct = deriveReduce(hidden, trc->rderivet, kernsz, n_part, latsz, n_derive, 0, trc->attsign, 1, "nc1");//[batch, kernel, party, latent, 1]
		if(nfinal) {//망호출 압축
			auto kfinal = reduct->slice({ { }, { kernsz - 1 } });//[batch, final_k, party, latent, 1]
			kfinal = kfinal->reshape({ -1, n_part, latsz });//[batch, party, latent], final_k와 derive는 1이되므로
			if(residual) {
				//kfinal = kfinal->layer_normal();
				//residual = residual->layer_normal();
				kfinal = *kfinal + *residual;//input or all residual
				//kfinal = kfinal->layer_normal();
			}
			if(prtbuild) { printf("nest chain call[batch, party, latent]\n"); kfinal->shape(); }
			kfinal = nestChain(kfinal, latsz, -1, "nest_call");//[batch, party, latent]
			kfinal = kfinal->transpose({ 0, 2, 1 });//[batch, latent, party], 모든정보가 혼합된 kfinal정보의 시퀀스를 입력 사이즈로 확장.
			stepcode = kfinal->layer_dense(seqsz, actfAnt, Initializer::xavier, "nc2");//[batch, latent, seqsz]
			stepcode = stepcode->transpose({ 0, 2, 1 });//[batch, seqsz, latent]
		} else {//파트가 4개 보다 적으면 더이상 망호출않고 현재까지의 얽혀진 것을 끝으로 확장.
			auto kfinal = hidden->slice({ { }, { kernsz - 1 } });//[batch, final_k, party, latent, derive]
			stepcode = deriveReduce(kfinal, 3, kernsz, n_part, latsz, n_derive, 0, trc->attsign, seqsz, "nc3");//[batch, seqsz, latent]
		}
		if(prtbuild) { printf("nest chain out[batch, seqsz, latent]\n"); stepcode->shape(); }
		if(resiopt > 1) {//output or all residual
			reduct = reduct->transpose({ 0, 2, 1, 3, 4 });//[batch, party, kernel, latent, 1]
			reduct = reduct->reshape({ -1, n_part * kernsz, latsz });//[batch, party * kernel, latent]
			if(n_part * kernsz != seqsz) reduct = reduct->slice({ { }, { 0, seqsz } });//[batch, seqsz, latent]
			//reduct = reduct->layer_normal();
			//stepcode = stepcode->layer_normal();
			stepcode = *reduct * *stepcode;//[batch, seqsz, latent], output or all residual
			//stepcode = stepcode->layer_normal();
		}
		stepcode = feed_forward(stepcode);
		trc->endscope();
		return stepcode;
	}
	Flux *blockChain(Flux *inp, intt latsz, intt kernsz = -1, const bytet *name = "block_chain")
	{
		Trace *trc = TRACER(tcrAnt);
		ubytet dat_t = inp->qType;
		intt noc = 0;
		if(kernsz < 0) kernsz = kernszAnt;

		trc->namescope(name);
		intt seqsz = inp->fshape[1], featsz = inp->fshape[2], n_derive, sz_head = 0;
		Flux *dexp;
		intt width = (kernsz < widthAnt ? kernsz : widthAnt);
		//가급적이면 시퀀스 길이를 커널 사이즈/스트라이드의 배수로 하여 두번째 케이스로 실행한다. 시퀀스와 커널이 정합되지 
		//않으면(스트라이드에 따라 아닐수도 있지먄, 예를 들어 시퀀스32, 커널8, 스트라이드6이면 스트라이드 4번째(옵셋 24)에서 
		//멈추어 제로패딩하지 않음) 제로페딩이 설정되므로 학습 효율이 떨어질것.
		floatt r_contig = (width < 8 && rExContigAnt < 0.7 ? 0.7 : rExContigAnt);//폭이작고 rate가 적으면 조합갯수가 0가 되므로 이때는 비율을 높힌다.
		Flux *a = flux(tcrAnt, { 1, width, featsz }, dat_t, constant);
		a->fill(1.0);
		auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
		n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - 스트라이드 단위 파생 시퀀스 갯수
		//expand_mask->shape();
		//expand_mask->printo(1,2);
		//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
		dexp = expand_batch2(inp, expand_mask, strideAnt, kernsz, zpadAnt, noc);//배치의 모든 시퀀스들의 파생 조합 시퀀스 생성
		intt n_bind = dexp->fshape[1];//한개 시퀀스의 총 파생 조합 시붠스 갯수
		intt n_part = n_bind / n_derive;//transform stride단위가 시퀀스 길이에 정수배로 맞아 떨어지게 해야함, 아니면 transform에서 zero padding설정해야함.
		if(prtbuild) printf("block chain seq sz: %d feat sz: %d stride: %d kernel: %d derive: %d party: %d\n", seqsz, featsz, strideAnt, kernsz, n_derive, n_part);
		if(trc->positional > 0) {
			if(trc->positional == 1) trc->positional = 0;//처음 한번만 포지셔널, 음수이면 이하 리지주얼 블럭 계속 포지셔널
			dexp = dexp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
			dexp = dexp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
			auto enpos = flux(trc, 3, &dexp->fshape[2], dexp->qType, constant);//[party, kernel, featsz]
			enpos->sinpos(n_part * kernsz);//[seq(party, kernel), featsz]
			dexp = *dexp + *enpos;//sinuosid positional, [batch, derive, party, kernel, featsz] = [batch, derive, party, kernel, featsz] + [positional(party, kernel), featsz]
			dexp = dexp->transpose({ 0, 2, 1, 3, 4 });//[batch, party, derive, kernel, featsz]
		} else {
			dexp = dexp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
		}
		Flux *hidden, *stepcode;
		if(sz_head) {
			intt nhead = n_derive / sz_head, i = 0;
			auto depart = dexp->split(nhead, 2);//[batch, party, derive(nhead*sz_head), kernel, featsz]
			vector<Flux *> hids;
			for(auto iter : *depart) {//[batch, party, sz_head, kernel, featsz]
				auto a = kerneling(iter, 1, 0, kernsz, latsz, featsz, n_part * sz_head, n_part, sz_head, i++);
				hids.push_back(a);//[batch, kernel, party, latent, sz_head]
			}
			hidden = concat(&hids, 4);//[batch, kernel, party, latent, derive(nhead*sz_head)]
		} else hidden = kerneling(dexp, 1, 0, kernsz, latsz, featsz, n_bind, n_part, n_derive, -1);
		//hidden - [batch, kernel, party, latent, derive]
		if(prtbuild) { printf("block chain[batch, kernel, party, latent, derive]\n"); hidden->shape(); }
		auto kfinal = hidden->slice({ { }, { kernsz - 1 } });//[batch, final_k, party, latent, derive]
		stepcode = deriveReduce(kfinal, 3, kernsz, n_part, latsz, n_derive, 0, trc->attsign, seqsz, "dr2");//[batch, seqsz, latent]
		if(prtbuild) { printf("block chain out[batch, seqsz, latent]\n"); stepcode->shape(); }
		if(trc->resiopt) {
			auto reduct = deriveReduce(hidden, trc->rderivet, kernsz, n_part, latsz, n_derive, 0, trc->attsign, 1, "dr1");//[batch, kernel, party, latent, 1]
			reduct = reduct->transpose({ 0, 2, 1, 3, 4 });//[batch, party, kernel, latent, 1]
			reduct = reduct->reshape({ -1, n_part * kernsz, latsz });//[batch, seqsz, latent]
			stepcode = *reduct + *stepcode;
			//stepcode = stepcode->layer_normal();
			stepcode = *inp * *stepcode;//[batch, seqsz, latent]
			//stepcode = stepcode->layer_normal();
		}
		stepcode = feed_forward(stepcode);
		trc->endscope();
		return stepcode;
	}
};
/*
Flux *expand_batch2(Flux *ins, Flux *mask, intt stride, intt width) //[batch, seq, in feat]
{
	intt n = ins->fshape[1];

	n = ins->fshape[1] / stride;
	if(ins->fshape[1] % stride == 0) n--;
	n *= stride;
	n = n + width - ins->fshape[1];
	//n -= width;
	//if(n % stride) n = ((n / stride + 1) * stride) - n;
	//else n = 0;
	if(n) {//width의 끝으로부터 시퀀스의 끝까지 구간이 스트라이드 사이즈 단위에 정합되지 않으면 마지막에 모자르는 구간을 제로
		Flux *pad = flux(tcrAnt, { ins->fshape[0], n, ins->fshape[2] }, ins->qType, constant);//패딩하여 더한다.
		ins = concat({ ins, pad }, 1);//[batch, (seq + pad), in feat]
	}
	vector<Flux *> fxl;
	for(intt i = 0, n = ins->fshape[1];i + width <= n; i += stride) {
		fxl.push_back(ins->slice({ {}, {i, i + width} }));
	}
	ins = stack(&fxl, 1);//[batch, devide_seq, width, feat]
	ins->printo(1, 2);
	ins->shape();
	ins = ins->expand_dims(2);//[batch, devide_seq, 1, width, feat]
	ins->printo(1, 2);
	ins->shape();
	printf("--------------1----------------\n");
	ins = *ins * *mask;//[batch, devide_seq, n_derive, width, feat]=[batch, devide_seq, 1, width, feat] * [n_derive, width, feat]
	ins->printo(1, 2);
	ins->shape();
	printf("--------------2----------------\n");
	//ins = ins->bypass("22\n");
	ins = ins->reshape({ -1, ins->fshape[1] * ins->fshape[2], ins->fshape[3], ins->fshape[4] });
	ins->printo(1, 2);//[batch, bindbatch(devide_seq*n_derive), width, feat]
	ins->shape();
	return ins;
}*/