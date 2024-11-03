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
	//agzo.zpad�� 1�� �ϸ� �Է� ��ŭ�� ���������ǹǷ� �Է��� Ŀ�� ����� ���յ��� ���� ��� ������ ��Ʈ���̵�
	//������ ���� ������ ���� ��Ʈ���̵� ���հ������� ���� ������ ������ ��� ó���� ���� ����. 3,4���� �Է� ���������� ���ڸ���
	//�κ��� ���� �е��Ͽ� �̷μ� ������ �����ϹǷ� �е����� ���� �������� �����ǹǷ� �Է��� �ְ�ɼ��ִ�. 2���� �� �Է°�������
	//������ �����ϰ� ������ ��Ʈ���̵忡�� ���ڸ��� ���� ���Ϻ�(��)�� �ܼ� �����е��ϹǷ� ������ ������ ��� 0�� �Ǿ� �н���������
	//���õɼ������Ƿ�(����ġ�� ���� ���� ������ �����Ƿ�) �Է°��� �ְ���� �ʴ´�. 
	//�׷��� 2���� expand batch�� �ƴ� combination �Լ��θ� ����ɼ������Ƿ� �߰����� ���Ǹ� �� ���Ϸ� �����ĵ��� �ʴ´�.
	//���� 2���� convolving�� �ƴϰ� ���յ��� �ʴ� ���̽��� �ɼ��� �־� ����Ѵ�. ���յǸ� �����е� �ʿ�����Ƿ� �������.
	void optAnet(intt lookup_sz = 8, intt step = 8, sytet zpad = 4, floatt rexc = 0.7, sytet dot_t = STRIDE_DOT, bool el = 0)
	{
		Trace *trc = TRACER(tcrAnt);

		kernszAnt = (trc->kernelSize < 0 ? lookup_sz : trc->kernelSize); //Ŀ�� ������ ������
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
		latszAnt = latent_sz;  //���� ���� ���� ��� ����, ���� 3�� ������ �н��� ������ ���� ��ħ.
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

		if(kernsz < widthAnt) {//Ŀ�� ����� �������� ������ ���� ���� ������ Ŀ�λ���� ������ ����
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
			cat_ins = stack({ ins, shift_ins }, 2);//seq����(1����)�Ʒ� �����ϳ��� �� �����(2����) ���� ������(1����)�� �Ѱ��� ���� ���λ����� (2)������ ����
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
		if(mul > 1) {//Ŀ�λ���� �������� ���̺��� ũ�� �������� ������ ���̸� Ŀ�λ���� ���ְ�
			/* //width horiz concat example//�� �����ŭ �������� ���� ������ �и��Ǵ� ������ ���δ�.
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
		//��� ���� ������δ� ����� ���� ������ expand�Ҽ������Ƿ� �е��� �Ѵ�. �е� �ɼ��� outer(tzpad == 4)�� �ƴϸ�
		//��� inner padding���� ó���Ѵ�.
		if(kernsz < widthAnt) {//Ŀ�� ����� �������� ������ ���� ���� ������ Ŀ�λ���� ������ ����
			width = kernsz;
			mul = 1;
		} else if(kernsz % widthAnt) throwFault(-1, "not aligned kernel\n");
		else {
			width = widthAnt;
			mul = kernsz / widthAnt;
		}
		if(tzpad == 4) {//outer zero padding. ������ 32, Ŀ��8, ��Ʈ���̵� 6 �ΰ�� ������ ��Ʈ���̵�� 30�̰� 
			nseq = ins->fshape[1];//������ ������ 32���� ���ڸ��� 6���� (8 - (32 - 30))�� �����е��Ѵ�.
			nstride = nseq / stride;
			if(nseq % stride == 0) nstride--;
			nrest = nstride * stride;
			nrest = nrest + width - nseq;
		} else {//inner zero padding. ������ 32, Ŀ��8, ��Ʈ���̵� 6 �ΰ�� ������ ��Ʈ���̵�� 24�̰�
			nseq = ins->fshape[1] - width;//24 + 8(Ŀ��) == 32 ������ ����� ���յǹǷ� �����е����� ��.
			nstride = nseq / stride;
			if(nseq % stride) {
				nstride++;
				nrest = nstride * stride + width - ins->fshape[1];
			} else nrest = 0;
		}
		if(TRACER(tcrAnt)->derivefeat == 2 && outsz > 0) {//�������� 8 ~ 32���� �̸� 32�� Ȯ���Ͽ� 
			if(ins->fshape[1] > widthAnt && ins->fshape[1] < widthAnt * 4) {//������ 4�� ���ϵǰ� �Ѵ�.
				outsz = 4;
				nseq = widthAnt * outsz;
				nrest = nseq - ins->fshape[1];
				nstride = nseq / stride;
				if(nseq % stride == 0) nstride--;
			}
		}
		if(nrest) {//width�� �����κ��� �������� ������ ������ ��Ʈ���̵� ������ ������ ���յ��� ������ �������� ���ڸ��� ������ ����
			Flux *pad = flux(tcrAnt, { ins->fshape[0], nrest, ins->fshape[2] }, ins->qType, constant);//�е��Ͽ� ���Ѵ�.
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
		if(mul > 1) {//Ŀ�λ���� �������� ����(widthAnt)���� ũ�� �������� ������ ���̸� Ŀ�λ���� ���ְ�
			/* //width horiz concat example//�� �����ŭ �������� ���� ������ �и��Ǵ� ����(party)�� ���δ�.
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
	//��Ʈ���̵� ������ ����ġ ����
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
	Flux *trippledot2(Flux *inp, Flux *wsizing) //���� ����
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
		if(trc->positional > 0 && trc->rderivet == 0) {//aetpr.�Է��� �����ųε��ְ� party�� �����Ͽ� ���ټ� ����
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
		if(ni_part) wcross->fill(1.0);//�Է� ��Ʈ������ Ŀ���� ������ ���ܸ� ���ǹǷ� �̰͸� 
		//���ټ� ���ϰ� �Է���Ʈ���� ������ ������ �ܼ� �������Ѵ�.(�����������Ƿ� �����ĵ��� �ʴ´�.)
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
		//���������� Ŀ�ν����� �ǳʶٴ� ���� �����ϹǷ� ��ġ�� �����ϱ����� ����ġ�� Ŀ�ν��ܺ��� ���� �Ѵ�.
		//��Ʈ���̵尣�� ��ġ�� wcross�� ��Ʈ���̵庰�� ���� �ְ� �̸� ��Ʈ���̵� ���� �Ͽ� Ȯ���Ѵ�.
		//wxh�� ��ġ �к� ����̹Ƿ� derive�� ���� Ŀ�θ��Ҷ� �� derive�и����� �����ɼ��ֵ��� iname�� �������Ѵ�.
		Flux *wxh, *whh, *wh, *bh, **hiddens, *hidden, *state;
		vector<Flux *> hid_cat;
		intt t = 0;//�ð���(Ŀ�λ�����)�� ���� ������ ��(Ŀ�γ�)���� �������Ƿ� �߰��� 0�Էµ���(�����е�����) ���õǾ� ���������� ��������.
		//inp == [batch, party, derive, kernel, featsz]
		if(pos_kern > 0) {
			if(pos_kern == 6) {//whh�� ��� Ŀ�� ���ܺ��� ����ġ�� �ξ� ��ġ �к��ϰ� wxh�� Ŀ�� ���ܺ��� �ξ� 
				if(prtbuild) printf("kernel expand 6\n");//party������ �����Ѵ�.
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
					else {//hidden - [batch, derive, latent],�� party�� ���� t���� �ʱⰪ ������~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~�����Ͽ� party���� �������� ���������ʰ� �Ѵ�.
					}
					auto x = input_t->dot(_wxh, { {2}, {0} });//[batch, derive, latent]=[batch, derive, featsz]*[featsz latent]
					auto h = hidden->dot(_whh, { {2}, {0} });//[batch, derive, latent] = [batch, derive, latent][latent, latent]
					auto a = *(*x + *h) + *_bh;//[batch, derive, latent]=[batch, derive, latent]+[batch, derive, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//�� Ŀ���� ������ ���� ���� ����
				}
			} else if(pos_kern == 5) {//wxh�� whh�� party���� �ξ� party������ �����Ѵ�.
				if(prtbuild) printf("kernel expand 5\n");//TRUE RIGHT. 3������ 0.001���� ���ϳ� �����ȴٴ� ���鿡�� �� ���� ������
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
					else {//hidden - [batch, derive, latent],�� party�� ���� t���� �ʱⰪ ������~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~�����Ͽ� party���� �������� ���������ʰ� �Ѵ�.
						_wh = unstacked_wh->at(t / kernsz);//[featsz+latent latent]
						_bh = unstacked_bh->at(t / kernsz);//[1 latent]
					}
					auto concat_x = concat({ input_t, hidden }, 2);//[batch, derive, (featsz + latent)] = [batch, derive, featsz](1) + [batch, derive, latent]
					auto a = *concat_x->dot(_wh, { {2}, {0} }) + *_bh;//[batch, derive, latent] = [batch, derive, (featsz + latent)][(featsz + latent), latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//�� Ŀ���� ������ ���� ���� ����
				}
			} else if(pos_kern == 4) {//62.��� Ŀ�� ���ܺ��� ����ġ�� �ξ� ��ġ �к�
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
					else {//�� party�� ���� t���� �ʱⰪ ������~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~�����Ͽ� party���� �������� ���������ʰ� �Ѵ�.
					}//hidden - [batch, derive, latent]
					//_wh = concat({ _wxh, _whh }, 0);//[(featsz + latent) latent] = [featsz latent] + [latent, latent]
					auto concat_x = concat({ input_t, hidden }, 2);//[batch, derive, (featsz + latent)] = [batch, derive, featsz](1) + [batch, derive, latent]
					auto a = *concat_x->dot(_wh, { {2}, {0} }) + *_bh;//[batch, derive, latent] = [batch, derive, (featsz + latent)][(featsz + latent), latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//�� Ŀ���� ������ ���� ���� ����
				}
			} else if(pos_kern == 3) {//TRUE RIGHT.wxh�� ��� Ŀ�� ���ܺ��� ����ġ�� �ξ� ��ġ �к��ϰ� whh�� party���� �ξ� 
				if(prtbuild) printf("kernel expand 3\n");//party������ �����Ѵ�.
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
					else {//hidden - [batch, derive, latent],�� party�� ���� t���� �ʱⰪ ������~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~�����Ͽ� party���� �������� ���������ʰ� �Ѵ�.
						_whh = unstacked_whh->at(t / kernsz);//[latent latent]
						_bh = unstacked_bh->at(t / kernsz);//[1 latent]
					}
					auto x = input_t->dot(_wxh, { {2}, {0} });//[batch, derive, latent]=[batch, derive, featsz]*[featsz latent]
					auto h = hidden->dot(_whh, { {2}, {0} });//[batch, derive, latent] = [batch, derive, latent][latent, latent]
					auto a = *(*x + *h) + *_bh;//[batch, derive, latent]=[batch, derive, latent]+[batch, derive, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//�� Ŀ���� ������ ���� ���� ����
				}
			} else if(pos_kern == 2) {//wxh�� Ŀ�� ���ܺ��� ����ġ�� �ΰ� Ŀ�γ������� ��ġ �к��ϰ� party������ �����ϰ�  
				if(prtbuild) printf("kernel expand 2\n");//whh�� �ϳ��� �ξ�  ��� �����Ѵ�.
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
					else {//hidden - [batch, derive, latent],�� party�� ���� t���� �ʱⰪ ������~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~�����Ͽ� party���� �������� ���������ʰ� �Ѵ�.
					}
					auto x = input_t->dot(_wxh, { {2}, {0} });//[batch, derive, latent]=[batch, derive, featsz]*[featsz latent]
					auto h = hidden->dot(whh, { {2}, {0} });//[batch, derive, latent] = [batch, derive, latent][latent, latent]
					auto a = *(*x + *h) + *bh;//[batch, derive, latent]=[batch, derive, latent]+[batch, derive, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//�� Ŀ���� ������ ���� ���� ����
				}
			} else {//wxh�� ��� Ŀ�� ���ܺ��� ����ġ�� �ξ� ��ġ �к��ϰ� whh�� �ϳ��� �ξ� 
				if(prtbuild) printf("kernel expand 1\n");//��� �����Ѵ�.
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
					else {//hidden - [batch, derive, latent],�� party�� ���� t���� �ʱⰪ ������~
						hidden = flux(tcrAnt, { inp->fshape[0], n_derive, latsz }, lat_tAnt, constant);
						hidden->fill(0.0, inp);//~�����Ͽ� party���� �������� ���������ʰ� �Ѵ�.
					}
					auto x = input_t->dot(_wxh, { {2}, {0} });//[batch, derive, latent]=[batch, derive, featsz]*[featsz latent]
					auto h = hidden->dot(whh, { {2}, {0} });//[batch, derive, latent] = [batch, derive, latent][latent, latent]
					auto a = *(*x + *h) + *bh;//[batch, derive, latent]=[batch, derive, latent]+[batch, derive, latent]+[1 latent]
					//if(prtbuild) { printf("enocde gate[batch, derive, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, derive, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
					else if(t && t % kernsz == 0) hid_cat.push_back(hiddens[t - 1]);//�� Ŀ���� ������ ���� ���� ����
				}
			}
			if(dual == 0) hid_cat.push_back(hiddens[t - 1]);//������ ���� ���� ����
			hidden = concat(&hid_cat, 1);
			if(dual) {//[batch, party * kernel * derive, latent]
				hidden = hidden->reshape({ -1, n_part, kernsz, n_derive, latsz });//[batch, party, kernel, derive, latent]
				if(derivefeat == 1) hidden = hidden->transpose({ 0, 2, 1, 3, 4 });//[batch, kernel, party, derive, latent]
				else if(derivefeat == 0) hidden = hidden->transpose({ 0, 2, 1, 4, 3 });//[batch, kernel, party, latent, derive]
			} else {//[batch, party * derive, latent]
				hidden = hidden->reshape({ -1, n_part, n_derive, latsz });//[batch, party, derive, latent]
			}
		} else {//����ġ�� party�ȿ��� Ŀ�� ���ܺ��� �ξ� ��ġ ����, party������ ����
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
					auto a = *concat_x->dot(_wh, { {2}, {0} }) + *_bh;//[batch, bindbatch, latent] = [batch, bindbatch, (featsz + latent)][(featsz + latent), latent], ���� bindbatch������� ������ ���ε� �������̴�.
					//if(prtbuild) { printf("enocde gate[batch, bindbatch, latent]\n");a->shape(); }
					hiddens[t] = a->actf(actf_code);//[batch, bindbatch, latent]
					if(dual) hid_cat.push_back(hiddens[t]);
				}
			} else if(pos_kern == -1) {//��� party�� Ŀ���� �����Ͽ� wxh�� Ŀ�� ���ܺ��� ����ġ�� 
				if(prtbuild) printf("kernel expand -1\n");//�ξ� ��ġ �к��ϰ� whh�� �ϳ��� �ξ� �����Ѵ�.
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
			} else if(pos_kern == -2) {//wxh, whh�� �ϳ��� �ξ� �����Ѵ�.
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
			} else if(pos_kern == -3) {//wxh�� �ϳ��� �ξ� �����Ѵ�. whh�� derive���� �д�.
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
			} else if(pos_kern == -4) {//wxh�� Ŀ�� ���ܺ��� ����ġ�� �ΰ� Ŀ�γ������� ��ġ �к��ϰ� 
				if(prtbuild) printf("kernel expand -4\n");//party������ �����ϰ�. whh�� derive���� �д�.
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
			} else if(pos_kern == -5) {//wxh�� �ϳ��� �ξ� �����Ѵ�. whh�� derive���� �д�. ���ʸ� ȣ���Ҷ� latsz�� 
				if(prtbuild) printf("kernel expand -5\n");//n_derive ������� ����ϰ� dual encoder���� �����־�
				latsz = 1;									//�ɼ�����. ��� ���� ����, �׽�Ʈ ��
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
	//ni_part - 0���� ������ �Ϲ� ���ڴ�, ���ڴ����� ȣ��, 0 �̻��̸� ������ڴ����� ȣ��
	//kernsz - ��� ���ڴ����� Ŀ�λ�����(8)�� ��� ���ܿ��� ����Ҷ� �ǹ�, ��� ���ڴ��� �ƴ� ������ 
	//ȣ��Ǹ� Ŀ���� ������ ���ܸ� ��µǹǷ� �� ���� 1�� ȣ��ȴ�.
	Flux *_deriveReduce(Flux *hidden, intt rderivet, intt kernsz, intt n_part, intt latsz,
		intt n_derive, intt ni_part, sytet attend, intt cross_out, const bytet *name = "derive_reduce")
	{
		Trace *trc = TRACER(tcrAnt);
		Flux *wcross, *wcross_b = nullx;
		//hidden - if(trc->derivefeat == 2) [batch, party, kernel, derive, latent]
		//			else [batch, kernel, party, latent, derive]
		//ni_part = 4;//��)psz=8, ni_part=4, batch=1, latent=32, cross_out=1
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
		} else if(rderivet == 2) {//61.derive reduce�� �� Ŀ�� Ÿ�Ӻ� ����ġ�� ���Ͽ� �Ѵ�.
			wcross = flux(tcrAnt, { kernsz*n_part, n_derive, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { kernsz*n_part, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
			if(ni_part >= 0) hidden = hidden->reshape({ -1, kernsz * n_part, latsz, n_derive });//[batch, kernel * party, latent, derive]
			if(prtbuild) { printf("derive_reducer derive reduce 2 [batch, kernel(%d) * party, latent, derive]\n", kernsz); hidden->shape(); }
			//[batch, kernel * party, latent, cross_out] = [batch, kernel * party, latent, derive] * [kernel * party, derive, cross_out] 
			hidden = stridedot(hidden, wcross, wcross_b);//[1,8*8,32,1]
			if(ni_part >= 0) hidden = hidden->reshape({ -1, kernsz, n_part, latsz, cross_out });//[batch, kernel, party, latent, cross_out]
		} else if(rderivet) {//TRUE RIGHT.derive reduce�� party�� ����ġ�� ���Ͽ� �Ѵ�.
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
		} else {//derive reduce�� ��� ���ܿ��� ����ġ�� �����Ͽ� ���Ѵ�.
			if(trc->poskern == 0 && ni_part >= 0) hidden = hidden->reshape({ -1, kernsz, n_part, latsz, n_derive });//[batch, kernel, party, latent, derive]
			if(prtbuild) { printf("derive_reducer derive reduce 0 attend %d [batch, kernel(%d), party, latent, derive]\n", attend, kernsz); hidden->shape(); }
			if(attend) {//derivefeat�� ���������� latent �� derive�� ��� �ٲ� �����̴�.
				hidden = hidden->reshape({ -1, latsz, n_derive });//[batch*kernel*party, latent, derive]
				hidden = attention_layer(hidden, cross_out);//[batch*kernel*party, latent, cross_out]
				if(ni_part >= 0) hidden = hidden->reshape({ -1, kernsz, n_part, latsz, cross_out });//[batch, kernel, party, latent, cross_out]
			} else {
				intt idot = (ni_part >= 0 ? 4 : 3);
				wcross = flux(tcrAnt, { n_derive, cross_out }, lat_tAnt, (trc->reducea ? constant : trainable), Initializer::xavier, "wcross");
				if(trc->reducea == 0) wcross_b = flux(tcrAnt, { cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
				//[batch, kernel, party, latent, cross_out] = [batch, kernel, party, latent, derive] * [derive, cross_out] 
				if(trc->reducea) {//latent���� derive�� ���Ѵ�.
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
		intt n_derive = emask->fshape[0];//��Ʈ���̵� ���� �Ļ� ������ ����
		if(trc->multihead == 1) sz_head = 1;
		else if(trc->multihead > 1) {
			sz_head = (n_derive / trc->multihead);
			if(sz_head <= 1) sz_head = 1;//sz_head�� 1�̸� n_derive�� 1�� ����ϰԵǹǷ� ��ȭ����.
			else if(n_derive % trc->multihead) {
				sz_head++;
				if(trc->outerMask) {//��󸣴� �� ���� �е�
					n_derive = (n_derive / sz_head) * sz_head + sz_head;
					auto a = flux(emask, constant);
					a->resizing4(n_derive - emask->fshape[0]);//emask�� ù°�� ���� ũ�⿡��   
					a->fill(0.0);		//n_derive�� ������ ���� �������� �����е� ����ũ ����
					emask = concat({ emask, a }, 0);//emask�� ù°�� ũ�⸦ n_derive�� ����
				} else {//�������� ����
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
		if(cmodAnt < 0) {//ù��° ���ڵ��̸� �����ǰ� ���Ŀ��� �� ������� 
			if(outsz > 0 && inp->fshape[1] > outsz) cmodAnt = C_ZZ;//�Է� ������ ���̺��� ��� ������ ���̰� ������ ������ ��� ���� ����.
			else {//������ ���� ����̰ų� outsz�� �����̸� ������ ������ ��Ҿ��� �ѹ��� Ÿ�� ����
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
		if(kernsz < 3) throwFault(-1, "seq size too small\n");//3���� ������ �ĺ���̼ǿ��� ���յ��� �������ִ�.
		if(trc->spotOpt == 1 && seqsz == kernsz) latsz = seqsz;
		//�������̸� ������ ���̸� Ŀ�� ������/��Ʈ���̵��� ����� �Ͽ� �ι�° ���̽��� �����Ѵ�. �������� Ŀ���� ���յ��� 
		//������(��Ʈ���̵忡 ���� �ƴҼ��� ������, ���� ��� ������32, Ŀ��8, ��Ʈ���̵�6�̸� ��Ʈ���̵� 4��°(�ɼ� 24)���� 
		//���߾� �����е����� ����) ��������� �����ǹǷ� �н� ȿ���� ��������.
		floatt r_contig = (width < 8 && rExContigAnt < 0.7 ? 0.7 : rExContigAnt);//�����۰� rate�� ������ ���հ����� 0�� �ǹǷ� �̶��� ������ ������.
		//floatt r_contig = (width < 8 ? 1 : rExContigAnt);
		if(trc->ebatch_t == 2) {
			Flux *a;
			a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			if(a->end_p() == nullx) throwFault(-1, "mask memory alloc fail\n");
			a->fill(1.0);
			auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - ��Ʈ���̵� ���� �Ļ� ������ ����
			//expand_mask->shape();
			//expand_mask->printo(1,2);
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			dexpend = expand_batch2(inp, expand_mask, strideAnt, kernsz, zpadAnt, outsz);//��ġ�� ��� ���������� �Ļ� ���� ������ ����
			//dexpend->printo(1, 2);
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, strideAnt, r_contig, zpadAnt, 0, 0);//üũ�� �ڵ�
			//if(a->fshape[0] != dexpend->fshape[1]) throwFault(-1, "check error\n");
		} else if(trc->ebatch_t || outsz % kernsz || seqsz % kernsz ||
			((floatt)kernsz / (floatt)strideAnt != 1.0) && ((floatt)kernsz / (floatt)strideAnt != 2.0)) {
			//�ĺ���̼� �Լ��� ���� stride�� ������ �Ҽ� �����Ƿ� ù��° encode���� ����Ѵ�. Ŀ�λ���� widthAnt����
			//ũ�� �ϸ� ������ �ʹ� �������Ƿ� �� �����Ϸ� �����ϴ�.
			dexpend = trc->tcr_combination2(inp, kernsz, strideAnt, r_contig, zpadAnt, 1);
			n_derive = tcrAnt->tcr_reserve;
			if(dexpend->fshape[1] % n_derive) throwFault(-1, "not aligned\n");//üũ�� �ڵ�
			//dexpend->printo(1, 2);//[batch, bindbatch(devide_seq+derive), width, feat]
			//dexpend->shape();
		} else {
			auto *a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			a->fill(1.0);
			auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - ��Ʈ���̵� ���� �Ļ� ������ ����
			//expand_mask->shape();
			//expand_mask->printo();
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			dexpend = expand_batch(inp, expand_mask, interv, kernsz);//��ġ�� ��� ���������� �Ļ� ���� ������ ����
			//dexpend->printo(1, 2);
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, stride, r_contig, zpadAnt, 0, 0);//üũ�� �ڵ�
			//if(a->fshape[0] != dexpend->fshape[1]) throwFault(-1, "check error\n");
		}//depand - [batch, bindbatch(party*derive), kernel, featsz]
		intt n_bind = dexpend->fshape[1];//�Ѱ� �������� �� �Ļ� ���� �Ôн� ����
		intt n_part = n_bind / n_derive;//transform stride������ ������ ���̿� ������� �¾� �������� �ؾ���, �ƴϸ� transform���� zero padding�����ؾ���.
		intt n_reduce = outsz / n_part;//Ŀ�� �����쳻 �Ļ� ���� ������ �������� ��ҵǴ� ��� ����
		if(n_reduce == 0) {//��� ����� ���� ��(�Է±��̸� Ŀ�α��̷� ���� ����) ���� ������, �� ������� ��°������� ������
			if(n_part % outsz == 0) {//��� ����� ���� ���� ������� ������, ���� ������ 4, ����� 2�̸�
				n_derive *= (n_part / outsz);//���� ������ ��� �������� ��� ������ ���̰� Ŀ�δ��� �Ļ����� �������� ������
				n_part = outsz;//�� ������ŭ �÷��� ��)���� ��»����� �������� ���� ���� ���Ϸ� �Ҽ��ְ� �Ѵ�.
			}//���Ұ����� ���հ����� �����Ҽ����� ��)���� ��»���� ���� ������ �°� �þ��.
			n_reduce = 1;//�ּ� ���Ҵ� 1�� ���Ϸ� ��ҵ� �� ����.
		}
		intt tarsz = outsz;
		outsz = n_part * n_reduce;//��.
		if(prtbuild) printf("encode gate seq sz: %d feat sz: %d tsz: %d stride: %d kernel: %d derive: %d party: %d reducing: %d outsz: %d\n", seqsz, featsz, tarsz, strideAnt, kernsz, n_derive, n_part, n_reduce, outsz);
		if(tcrAnt->dbgStep == 1) dexpend = dexpend->bypass("111\n");
		if(indiscret > 0) {//�Է� �Ӻ��� #inp == [batch, bindbatch, kernel, 1]
			if(dexpend->fshape[dexpend->fdim - 1] == 1) inp = dexpend->squeeze(dexpend->fdim - 1);//[batch, bindbatch(party*derive), kernel]//tf.reshape(self.input_data, [-1, self.input_data.shape[1], self.input_data.shape[2]])
			else inp = dexpend;//[batch, bindbatch(party*derive), kernel, featsz]
			inp = embedding(inp, indiscret, embedim, 1);//[batch, bindbatch, kernel, featsz(embedim)]
			if(prtbuild) printf("embedding inp [batch, bindbatch, kernel, featsz(embedim)] "); inp->shape();
			//�Է� ���İ� ���������̸� �� ���ĸ� �Ӻ����Ŀ� ���ĸ� �Ѱ��� ���Ѵ�.
			if(inp->fdim > 4) inp = inp->reshape({ inp->fshape[0], inp->fshape[1], inp->fshape[2], -1 });
			featsz = inp->fshape[inp->fdim - 1];//embedim
			//inp = inp->bypass("111-2\n");
			//if(featsz != embedim) throwFault(-1, "embed size\n");
			if(trc->positional > 0) {
				if(trc->positional == 1) trc->positional = 0;//ó�� �ѹ��� �����ų�, �����̸� ���� �����־� �� ��� �����ų�
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
		if(sz_head) {//derive�� �����Ͽ� Ŀ�θ� ����.
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
		//���� ������ ������ ���������� ���ε� ��ġ�� �Ѱ� ������ ������� ��ȯ
		Flux *outtens, *znode, *encode;
		intt cross_out;
		if(cmodAnt == C_XT) cross_out = outsz;// #�ѹ��� ��� ����
		else cross_out = n_reduce;//��ä�ڵ� �߻�ȭ ����
		if(trc->reducea) {
			dot_t = ORTHO_DOT;
			trc->ortho_single = 1;
			trc->attsign = 0;
		}
		if(AttentionOrOnlyFinalEncodingAttentionOpt(trc, n_part, spot_att)) goto NO_WEIGHT;//���ټ� �����̸� ���ο��� ����ġ �����ϹǷ� ���� ����ġ �ʿ����.
		if(dot_t == ORTHO_DOT || dot_t == STRIDE_DOT || trc->derivefeat == 2) goto NO_WEIGHT;
		if(dot_t == TRIPPLE_DOT) {//���� ������ ����ġ ����.(Ŀ�� ����������� ������ ����)
			wcross = flux(tcrAnt, { n_derive, latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { n_derive, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		} else if(dot_t == COUPLE_DOT) {//��� �Ļ� ���պ��� ���� ����ġ(���ü� ���� �ǹ�)
CDOT_LA:;	wcross = flux(tcrAnt, { n_bind, latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { n_bind, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		} else {//TENSOR_DOT, ��� �Ļ� ������ ����ġ ����(������ ���� ���� ����)
			wcross = flux(tcrAnt, { latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		}
NO_WEIGHT:;
		if(dot_t == ORTHO_DOT && trc->derivefeat != 2) {//����(����)�� �ĺ����(CNN)�� ����(����)�� ������. derive�� feature map
			if(trc->spotOpt == 2 && n_part == 1) {//������ Ŀ�λ����� ���̸� ���ڵ��ϴ� ���̰� �����ɼ��̸� ���纰�� derive�� ���Ѵ�.
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
			} else if(trc->ortho_single) {//���Ұ��� ����ġ ����, ��Ʈ���̰��� ������ ����, �Է±���8->���1�� ���� ������ŭ ������ ȣ���ϴ� ���� �Ѳ����� �����ϴ� �Ͱ� ����
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
				goto CDOT_LA2;//������ Ŀ�λ����� ���̸� ���ڵ��ϴ� ���̰� �����ɼ��̸� derive���� ������ ���Ѵ�.
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
CDOT_LA2:;	//ORTHO_DOT���� �°Ÿ�[batch, latent, cross_out] = [batch, latent, bindbatch] * [latent, bindbatch, cross_out]
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
			encode = znode->reshape({ -1, outsz, znode->fshape[3] });//[batch, outsz(party*reducing), derive/latent(ORTHO_DOT)], �����ڵ�� ������ ������� ����
		} else encode = znode;//[batch, party, latent]
		if(n_part == 1 && dot_t == ORTHO_DOT && encode->fshape[1] == residual->fshape[1]) encode = *encode + *residual;
		encodeOut = encode;//stratus���� ���
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
		if(cmodAnt == C_XT) {//inp == [batch, bindbatch, cross_out]//ageg)�� ������Ͽ� ��� 
			/*if(elastic) {//depricate	//������ ����[batch, outsz, lattent]�� �ٷ� ��� �ϱ⶧����
				intt n_bind = inp->fshape[1];	//���ڴ��� ��ŵ�Ѵ�.
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
		} else {//���ڵ� ������ ����� ���ڵ� ������ ������� Ȯ��, inp == [batch, insz(party*reducing), derive/latent(ORTHO_DOT)]
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
				}//else ���ڴ����� ���ټ� ���� �ɼ��̸� ����ġ �����ϹǷ� ���� ����ġ �ʿ����.
			} else if(szmhead) {//TRUE RIGHT.dot_t�� STRIDE_DOT�϶��� �����ǰ� �ǹ����� - depricate szmhead���� ���� ����
				nhead = n_derive / szmhead;
				wseq = flux(tcrAnt, { nhead, insz, outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq");
				wseq_b = flux(tcrAnt, { nhead, outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq_b");
				if(prtbuild) printf("decode gate multi-head\n");
			} else {
				wseq = flux(tcrAnt, { n_derive, insz, outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq");
				wseq_b = flux(tcrAnt, { n_derive, outsz }, lat_tAnt, trainable, Initializer::xavier, "wseq_b");
			}
			znode = inp->transpose({ 0, 2, 1 });//[batch, derive/latent(ORTHO_DOT), insz(party*reducing)], derive���� interval, interval���̸� �߷��ϰԵǾ� ��������� ����.
			if(dot_t == TENSOR_DOT || dot_t == TRIPPLE_DOT) {
				//[batch, derive, outsz] = [batch, derive/latent(ORTHO_DOT), insz]*[insz. outsz]
				if(decatt) {//ORTHO_DOT�� �ƴϾ derive�̸� ����� �޸� �Ҵ緮�� �ʹ� Ŀ �Ҵ� �������� ORTHO_DOT�϶���
					znode = attention_layer(znode, outsz)->actf(actfAnt);//latent�� 128�� ������ ���������� �Ҵ翡�� ���Ƿ� ����.
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
		/*nxcicp.�������� �ɼ��̶� ���ڵ�2 �ɼ��̸� ���ڵ����� ������������ �����Ƿ� �ʿ����.
		if(inp->fshape[1] == kernsz && kernsz != outsz) { // && finalReduced) {//���� �����Ŀ� ù��° ���ڵ������� �������� ��ġ�Ѵ�.�ȱ׷���
			inp = inp->transpose({ 0, 2, 1 });//[batch, derive, insz(party*reducing)]//���������� �����ϰ� �ݺ��ϴ� ���� �ǹǷ�
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
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - ��Ʈ���̵� ���� �Ļ� ������ ����
			//expand_mask->shape();
			//expand_mask->printo(1,2);
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			dexpend = expand_batch2(inp, expand_mask, strideAnt, kernsz, zpadAnt, outsz);//��ġ�� ��� ���������� �Ļ� ���� ������ ����
			//dexpend->printo(1, 2);
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, strideAnt, r_contig, zpadAnt, 0, 0);//üũ�� �ڵ�
			//if(a->fshape[0] != dexpend->fshape[1]) throwFault(-1, "check error\n");
		} else {
			auto *a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			a->fill(1.0);
			auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - ��Ʈ���̵� ���� �Ļ� ������ ����
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			dexpend = expand_batch(inp, expand_mask, interv, kernsz);//��ġ�� ��� ���������� �Ļ� ���� ������ ����
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, stride, r_contig, zpadAnt, 0, 0);//üũ�� �ڵ�
			//if(a->fshape[0] != dexpend->fshape[1]) throwFault(-1, "check error\n");
		}
		if(tcrAnt->dbgStep == 1) dexpend = dexpend->bypass("555\n");
		intt n_bind = dexpend->fshape[1];//�Ѱ� �������� �� ���� ���� �Ôн� ����
		intt n_part = n_bind / n_derive;
		intt cross_out = outsz / n_part;//Ŀ�� �����쳻 �Ļ� ���� ������ �������� ��ҵǴ� ��� ����
		if(cross_out == 0) cross_out = 1;//�ּ� ��Ʈ���̵�� 1�� ���Ϸ� �� �� ����.
		intt tarsz = outsz;
		outsz = n_part * cross_out;
		if(prtbuild) printf("decode gate2 seq sz: %d tsz: %d stride: %d sz_kernel: %d derive: %d party: %d reducing: %d outsz: %d\n", seqsz, tarsz, strideAnt, kernsz, n_derive, n_part, cross_out, outsz);
		//with tf.variable_scope(name) :
		if(trc->spotOpt == 1 && seqsz == kernsz) latsz = outsz;
		inp = dexpend;//inp == [batch, bindbatch(party*derive), kernel, featsz]
		Flux *wcross = nullx, *wcross_b = nullx, *hidden;
		inp = inp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
		if(sz_head) {//derive�� �����Ͽ� Ŀ�θ� ����.
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
		//���� ������ ������ ���������� ���ε� ��ġ�� �Ѱ� ������ ������� ��ȯ
		if(decatt || trc->derivefeat == 2) goto NO_WEIGHT;//���ټ� �����̸� ���ο��� ����ġ �����ϹǷ� ���� ����ġ �ʿ����.
		if(dot_t == ORTHO_DOT || dot_t == STRIDE_DOT) goto NO_WEIGHT;
		if(dot_t == TRIPPLE_DOT) {//��Ʈ���̵� ������ ����ġ ����.(Ŀ�� ����������� ������ ����)
			wcross = flux(tcrAnt, { n_derive, latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { n_derive, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		} else if(dot_t == COUPLE_DOT) {//��� �Ļ� ���պ��� ���� ����ġ(���ü� ���� �ǹ�)
CDOT_LA:;	wcross = flux(tcrAnt, { n_bind, latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { n_bind, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		} else {//TENSOR_DOT, ��� �Ļ� ������ ����ġ ����(������ ���� ���� ����)
			wcross = flux(tcrAnt, { latsz, cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross");
			wcross_b = flux(tcrAnt, { cross_out }, lat_tAnt, trainable, Initializer::xavier, "wcross_b");
		}
NO_WEIGHT:;
		Flux *outtens, *znode, *encode;
		if(dot_t == ORTHO_DOT && trc->derivefeat != 2) {
			if(trc->spotOpt == 2 && n_part == 1) {//������ Ŀ�λ����� ���̸� ���ڵ��ϴ� ���̰� �����ɼ��̸� ���纰�� derive�� ���Ѵ�.
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
			} else if(trc->ortho_single) {//��Ʈ���̵尣�� ����ġ ����, ��Ʈ���̰��� ������ ����
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
				goto CDOT_LA2;//������ Ŀ�λ����� ���̸� ���ڵ��ϴ� ���̰� �����ɼ��̸� derive���� ������ ���Ѵ�.
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
CDOT_LA2:;	//ORTHO_DOT���� �°Ÿ�[batch, latent, cross_out] = [batch, latent, bindbatch] * [latent, bindbatch, cross_out]
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
			encode = znode->reshape({ -1, outsz, znode->fshape[3] });//[batch, outsz(party*reducing), derive/latent(ORTHO_DOT)], �����ڵ�� ������ ������� ����
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
	Flux *outGate(Flux *inp, intt ydisc, intt out_feat, Flux *&logits, const bytet *name = "anet_out", bool reuse = 0)//��� �������� ������� Ÿ�� ��������� ��ȯ.
	{
		Flux *pred;

		tcrAnt->namescope(name, reuse);
		if(ydisc) yDiscrete = ydisc;//��� ���İ� vocabulary size
		else {
			yDiscrete = 0;
			ydisc = 1;
		}
		auto hidden = inp->fshape[2];//inp == [batch, y_sz, bindbatch(latent)]
		//with tf.variable_scope(name) :
		if(yDiscrete == 0 && hidden == out_feat) logits = inp;
		else {//out_feat�� �Ϲ������� ���� ���� ��� ��ū���̵� 1�� �̴�.
			intt mo_feat = out_feat * ydisc;//��� ���� ������ 1���� ũ�� �� ������ŭ ���� ������ ��
			auto wod = flux(tcrAnt, { hidden, mo_feat }, lat_tAnt, trainable, Initializer::xavier, "wod");//weight output demension
			auto wod_b = flux(tcrAnt, { mo_feat }, lat_tAnt, trainable, Initializer::xavier, "wod_b");
			//yDiscrete�̸� [batch, out_seq, (out_feat)*vocab_sz] = [batch, out_seq, latent][latent, (out_feat)*vocab_sz]
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
			pred = logits->argmax(-1);//[batch, out_seq, (out_feat)], �����ڵ带 �ҽ� �Ǵ� Ÿ�� ���̵�� ��ȯ
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
			else if(label->fshape[label->fdim -1] == 1) s_label = label->squeeze(2);//���� 1�� ������ ���ش�.
			else if(label->fshape[label->fdim - 1] != logit->fshape[logit->fdim - 2]) {//���� ���İ� ������ 1�� �ƴѰ��
				//logit - [batch, out_seq, out_feat, vocab_sz], label - [batch, out_seq, out_feat]
				throwFault(-1, "logit: and label dimension inconstant: %d %d\n", 
					logit->fshape[logit->fdim - 2], label->fshape[label->fdim - 1]);
			} else s_label = label;//[batch, out_seq, out_feat]
			//s_label = s_label->bypass("lllllllllllllllllllllllll");
			auto label_ohe = s_label->one_hot(yDiscrete);//[batch, out_seq, (out_feat), vocab_sz]
			//label_ohe = self.label_smoothing(label_ohe, logit.shape[2]);
			if(prtbuild) { printf("calc loss logit & label[batch, out_seq, (out_feat), vocab_sz]\n"); logit->shape(); label_ohe->shape(); }
			logit = logit->softmaxCrossEntropy(label_ohe);
			if(((Trace *)tcrAnt)->batchloss) batchLoss = logit->mean(1);//��ġ���� ũ�ν���Ʈ���� ���� ��� ���
			else batchLoss = nullx;
			return logit->mean();//��ü ũ�ν���Ʈ���� ���� ���.
		} else {
			if(prtbuild) { printf("calc loss[batch, out_seq, out_feat]\n"); logit->shape();}
			if(((Trace *)tcrAnt)->batchloss) {
				logit = logit->meanSquareError(label, 0);//��ġ���� �������� ���
				batchLoss = logit->mean(1);//��ġ���� �������� ��� ���
				return logit->mean();//��ü �������� ��� ���
			} else {
				batchLoss = nullx;
				return logit->meanSquareError(label);//logit - [batch, outsz, out_feat]
			}
		}
	}
	Flux *calcAccuracy(Flux *pred, Flux *target, intt discrete_out = -1)
	{
		if(discrete_out < 0) discrete_out = yDiscrete;

		if(discrete_out) {//pred, traget - [batch, out_seq, 1], Ÿ�� ������Ʈ �߿��� zero�е尡 �ƴ� ���� ���� �c�� �п�
			auto target_not_pad = target->not_equal(0.0);//Ÿ�� ���޸�Ʈ�� ���� ���� ���� ������Ʈ�� ���� ���� ���� ���
			auto acc = (*(*pred->equal(target) * *target_not_pad) / *target_not_pad->sum())->sum();
			return acc;//1.0�̸� ��Ȯ�� 100%
		} else {//pred - [batch, out_seq, vocab_sz]
			return target->squaredDifference(pred)->mean()->sqrt();//���� ���� ��� ��Ʈ, 0.0�̸� ��Ȯ�� 100%
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
		intt save_latsz = latsz;//derivefeat�̸� �ؿ��� derive���� ���� ���� �ٲ�Ƿ� �� �� ����
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
			intt sz = n_derive;//�ٷιؿ��� derive�� ����Ͽ� cross_out���� ����ϴ� �������� �ڵ尡���ְ�
			n_derive = latsz;//�� ���̽��� latent�� ����ϴ� ����̹Ƿ� ���� �ٲ� ���� latent�� derive
			latsz = sz;//�� ���� �ٲ� ������ �����Ѵ�.(�ٷ� �ؿ��� latent�� �� ���� �������� ��ġ�ȴ�.)
			if(prtbuild) { printf("dual enocder layer derive feat:%d lattent: %d inversed [batch, party, derive, kernel, featsz]\n", latsz, n_derive); inp->shape(); }
		} else {//TRUE RIGHT. hidden - [batch, kernel, party, latent, derive]
			if(prtbuild) { printf("dual enocder layer[batch, party, derive, kernel, featsz]\n"); inp->shape(); }
		}
		if(trc->inplus) {//��Ʈ���̵带 Ŀ�λ���� ��ø�ǰ� �Ұ�� �̿ɼ����������Ѵ�. expand�� ������
			ni_part = 0;//�Է� ��Ʈ�� Ÿ����Ʈ�� �и��ϴ� ���� �Ұ��� �ϹǷ�
			//�Է�(������ ����)�� �����ϴ� ���̰���Ʈ ��ä�� ��ǥ������ �н��ǵ��� 0�� �����Ͽ� �������ڵ� 
			//�н� �ǵ��� �Ѵ�. �Է°��� ���� �����ڵ�(pcode)�� �ϴ� ���� ������ 0�̹Ƿ� �ǹ̾���. 
			//���̰���Ʈ�� [|input] + <go token> + target�� �����ϰ�
			//��ǥ������ [|input] + target + <end token>�� �����Ѵ�.
			//���� pre-train�ҰŸ� ���̰���Ʈ�� �����ϰ� ��ǥ���� �����Ѵ�.
		}
		Flux *outtens = deriveReduce(hidden, trc->rderivet, kernsz, n_part, latsz, n_derive, ni_part, trc->attsign, cross_out);
		if(trc->derivefeat == 2) outtens = outtens->expand_dims(3);//[batch, party, kernel, 1, latent]
		else outtens = outtens->transpose({ 0, 2, 1, 4, 3 });//[batch, party, kernel, cross_out, latent]
		if(prtbuild) { printf("dual enocder tstep derive reduce[batch, party, kernel, cross_out, latent]\n"); outtens->shape(); }
		if(n_part != outtens->fshape[1] || latsz != outtens->fshape[4] || cross_out != outtens->fshape[3]) exit(1);//üũ�� �ڵ�
		vector<Flux *> _by_cat, *by_cat, t_cat;
		Flux *kfinal, *kstep;
		Flux *zpad = flux(tcrAnt, { outtens->fshape[0], outtens->fshape[3], outtens->fshape[4] }, lat_tAnt, constant);
		zpad->adjust(inp);//[1,1,32]
		zpad->fill(0.0);
		zpad = zpad->reshape({ -1, 1, 1, cross_out, latsz });//[batch, p, k, cross_out, latent][1,1,1,1,32]
		if(psz) {//�������� ����Ǿ� ���ԵǴ� �ڵ尡������ ����
			if(pcode->fshape[2] != latsz) {//���� �����ڵ� ������ ���������� ������ �����.
				//(latsz�� ������ inverse�Ǿ� derive size��)
				pcode = pcode->layer_dense(latsz, actfAnt, Initializer::xavier, "pc_inverse");
			}
			pcode = pcode->reshape({ -1, 1, 1, psz, latsz });//[batch, p, k, psz, latent][1,1,1,8,32]
		}//outtens - [batch, party, kernel, cross_out, latent]
		if(ni_part) {//�Է���Ʈ ��Ʈ���̵尡 ������ �� ������ŭ �� ��Ʈ���̵� Ŀ���� ������ k ��°�� �����Ͽ� ����
			kfinal = outtens->slice({ { }, { 0, ni_part }, { kernsz - 1 } });//��)[1,4,1,1,32]
			by_cat = kfinal->split(ni_part, 1);//ni_part * [batch, p, k, cross_out, latent] 4 * [1,1,1,1,32]
		} else by_cat = &_by_cat;
		for(intt i = ni_part; i < n_part; i++) {//Ÿ�� ��Ʈ���̵� ���� * Ŀ�� ������ �������� �Ѱ� �ڵ����
			for(intt j = 0; j < kernsz; j++) {
				if(psz) t_cat.push_back(pcode);//[batch, p, k, psz, latent][1,1,1,8,32]
				if(ni_part || i > ni_part) t_cat.insert(t_cat.end(), by_cat->begin(), by_cat->end());//4 * [1,1,1,1,32]
				kstep = outtens->slice({ { }, { i }, { j } });//[batch, p, k, cross_out, latent][1,1,1,1,32]
				t_cat.push_back(kstep);//�� ��Ʈ���̵� �� Ŀ�ν����� ���� ����
				for(intt k = i + 1; k < n_part; k++) t_cat.push_back(zpad);//���� ��Ʈ���̵��� ������ ���� �е�[1,1,1,1,32]
			}
			kfinal = outtens->slice({ { }, { i }, { kernsz - 1 } });//[batch, p, k, cross_out, latent][1,1,1,1,32]
			by_cat->push_back(kfinal);//���� ��Ʈ���̵�� �̹� ��Ʈ���̵��� ������ Ŀ�ν��� �����(����)�� �����ϰ� �Ѵ�.
		}
		intt nt_part = n_part - ni_part;//Ÿ����Ʈ ��Ʈ���̵� ����
		//nt_part * kernel * (pcode + party) * [batch, p, k, cross_out, latent]4 * 8 * (8 + 8) * [1,1,1,1,32]
		stepcode = concat(&t_cat, 3);//[batch, p, k, nt_part * kernel * (psz + party), cross_out, latent][1,1,1,4*8*(8+8),1,32]
		intt zsz = 1, sz_seq = (psz + n_part) * cross_out;//(psz + party) * cross_out
		if(sz_seq < 8 && trc->zstep == 0) {//reduce�� ������ ���̰� 7�����̸� �Ļ����� ������
			trc->zstep = 3;//���� ��ȣ�� ��� ����� �������ϹǷ�(���� ��ȣ���ҰŸ� �̰��� -1��
		}	//�ϰ� derivefeat�� 0�μ���, ê������16���� ������) ��ȣ�⿡ ���� ������� ���� 4�����̽��� ��ҽ�Ų��.
		if(trc->zstep == 3) {//60.���ܺ� ����ġ�� �����Ͽ� ���� �ڵ� ���
			stepcode = stepcode->reshape({ -1, nt_part * kernsz, sz_seq, latsz });
			stepcode = stepcode->transpose({ 0, 1, 3, 2 });//[batch, nt_part * kernel, latent, sz_seq]

			Flux *wdstep = flux(tcrAnt, { nt_part * kernsz, sz_seq, 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep");
			Flux *wdstep_b = flux(tcrAnt, { nt_part * kernsz, 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep_b");
			//[batch, nt_part * kernel, latent, 1] = [batch, nt_part * kernel, latent, sz_seq] * [nt_part * kernsz, sz_seq, 1] + [party, sizing]
			stepcode = stridedot(stepcode, wdstep, wdstep_b);
			//stepcode = stepcode->reshape({ -1, nt_part, kernsz, 1, latsz });//[batch, nt_part, kernel, 1, latent][1,4,8,1,32]
			stepcode = stepcode->actf(actfAnt);//�׷��� nan�߻��Ǵ� ����־� ���ʿ�, �Ƹ��� ���߿� �޸� �뷮�� �Ǹ� derive reduce�� ���ټ����� �ؾ� �ҵ�, �������Ŀ��� ���� ���� ���� �������� Ȱ���Լ��� �ִ´�.
			if(prtbuild) printf("dual enocder tstep reduce 4\n");
		} else if(trc->zstep == 2) {//60.Ÿ�� ��Ʈ�� ����ġ�� �����Ͽ� ���� �ڵ� ���
			stepcode = stepcode->reshape({ -1, nt_part, kernsz, sz_seq, latsz });
			stepcode = stepcode->transpose({ 0, 2, 1, 4, 3 });//[batch, kernel, nt_part, latent, sz_seq]

			Flux *wdstep = flux(tcrAnt, { nt_part, sz_seq, 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep");
			Flux *wdstep_b = flux(tcrAnt, { nt_part, 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep_b");
			//[batch, kernel, nt_part, latent, 1] = [batch, kernel, nt_part, latent, sz_seq] * [nt_part, sz_seq, 1] + [nt_part, sizing]
			stepcode = dual_stridedot(stepcode, wdstep, wdstep_b);
			stepcode = stepcode->transpose({ 0, 2, 1, 4, 3 });//[batch, nt_part, kernel, 1, latent][1,4,8,1,32]
			stepcode = stepcode->actf(actfAnt);//�׷��� nan�߻��Ǵ� ����־� ���ʿ�, �������Ŀ��� ���� ���� ���� �������� Ȱ���Լ��� �ִ´�.
			if(prtbuild) printf("dual enocder tstep reduce 3\n");
		} else if(trc->zstep == 1) {//��� ������ ����ġ�� �����Ͽ� ���� �ڵ� ���
			stepcode = stepcode->reshape({ -1, nt_part, kernsz, sz_seq, latsz });
			stepcode = stepcode->transpose({ 0, 1, 2, 4, 3 });//[batch, nt_part, kernel, latent, sz_seq]

			Flux *wdstep = flux(tcrAnt, { sz_seq, 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep");
			Flux *wdstep_b = flux(tcrAnt, { 1 }, lat_tAnt, trainable, Initializer::xavier, "wdstep_b");
			//[batch, nt_part, kernel, latent, 1] = [batch, nt_part, kernel, latent, sz_seq] * [sz_seq, 1] + [1]
			stepcode = *stepcode->dot(wdstep, { {4}, {0} }) + *wdstep_b;
			//stepcode = stepcode->reshape({ -1, nt_part, kernsz, 1, latsz });//[batch, nt_part, kernel, 1, latent][1,4,8,1,32]
			stepcode = stepcode->actf(actfAnt);//�׷��� nan�߻��Ǵ� ����־� ���ʿ�, �������Ŀ��� ���� ���� ���� �������� Ȱ���Լ��� �ִ´�.
			if(prtbuild) printf("dual enocder tstep reduce 2\n");
		} else {//trc->zstep == 0, //TRUE RIGHT.����� ����
			stepcode = stepcode->reshape({ -1, sz_seq, latsz });//[batch * nt_part * kernel, sz_seq, latent][1*4*8,(8+8)*1,32]
			if(stepcode->fshape[1] > 3) {//�� Ÿ���� ������ ������ ���̰� 4�̻��̸� ������ ����
				//dualEncoderLayer�� �����־�� �ݺ��ǹǷ� ���⼭ �缳���Ǵ� ��� trc������ ������Ų��.
				floatt low_bound = trc->lowbound;
				intt convolv = trc->convolving, stride_jump = trc->strideJump, kern_size = trc->kernelSize;
				sytet dot_type = trc->dotType, pos_kern = trc->poskern, r_drive = trc->rderivet;
				sytet enc_kern = trc->enckern;
				//��Ʈ���̵� ������ ���� ������ �����Ƿ� t step�� t�ð������� �� ��Ʈ���̵��� ������ t��������
				//generic���� ���� ���� �����ڵ� ����, ��Ʈ���̵尡 8�� �̻��̸� zcode�� 2���̻� ������� �ƴ�
				trc->lowbound = 0.125;//lower bound, ���� ���Ѽ��� 8 Ŀ�λ����� �� 1 �� ����, �̰��� 0.25��
				//�Ͽ� 8 Ŀ�λ����� �� 2�� �Ǿ� ���������� ��»���� 1�εǰ� �ص� ��»���� �ּ� 2���ϰ� �ɼ�����.
				trc->convolving = 8;//������, �Է��� �� ��������� ��¾����ڵ带 1������� ���, �Է±��̰� �� 
				//����� ������ 8Ŀ�δ����� 1�����̾� ����, ��)�Է±��� 8->1, 16->2, �������� 16�̸� 
				//�Է±��� 8->1, 16->1, 32->2, �������� 32�̸� �Է±��� 8->1, 16->1, 32->1, 64->2 
				if(trc->derivefeat) trc->dotType = STRIDE_DOT;
				else trc->dotType = ORTHO_DOT;//����(����)�� �ĺ����(CNN)�� ����(����)�� ������.
				if(trc->positional > 0) {
					if(trc->poskern == 4) trc->poskern = 2;//�Է� ����������
					//wxh����ġ�� ���� �Ͽ� ��ġ���� ���ϰ� ��� �����ų��ϴ� �ɼ��̸� �Ʒ� �� ȣ�⿡���� 
					//discrete������ �ƴϹǷ� �����ų��� ������� �����Ƿ� wxh����ġ�� ��ġ�����ϴ� �ɼǽ����Ѵ�.
				
				} else if(trc->positional < 0) trc->poskern = 6;
				if(trc->rderivet == 0) trc->rderivet = 1;//�� ����� ���� ����� �Ʒ� ������ �����Ŵ����� �����Ƿ� aetpr)���� party��
				//�������̰� ���ټ� ���� ����ɰ�쿡 party�� �����Ͽ� ���ټ� ���� ������� �ʵ��� 0�� �ƴ� ���ǰ� ����.
				trc->strideJump = -1;//�ʱ� �Է¸� ������ ��Ʈ���̵�� �ϰ� �Ʒ��������� �⺻ ��Ʈ���̵�(width����)���� �ǵ��� ����.
				if(trc->szkernDualNest) trc->kernelSize = trc->szkernDualNest;
				if(trc->enckern != 100) trc->enckern = 100;//enckern reset, ������ڴ����� ȣ��Ǵ� �ڵ���� ���ڵ���
				if(prtbuild) {								//enckern�� ������� �ʰ� �Ѵ�. 
					printf("dual enocder tstep reduce 0 conv: 8 lbound: 0.125 nt_part: %d kernel: %d psz: %d party: %d reducing: %d [batch * nt_part * kernel, (psz + party) * reducing, latent]\n",
						nt_part, kernsz, psz, n_part, cross_out); stepcode->shape();
				}
				trc->nbyout = -1;//dual incoder off//stepcode - [batch * nt_part * kernel, sz_seq, latent]
				Generic *seqout = generic(stepcode, nullx, save_latsz, 0, 0, 0, -1, -1, "dual_seqout");
				stepcode = seqout->zcodec;//[batch * nt_part * kernel, zcode, latent][1*4*8,z,32]
				latsz = stepcode->fshape[2];//derivefeat�� ��� �������� 8�� �ȵǸ� �Ļ����� �������Ƿ� �缳��.
				trc->lowbound = low_bound; trc->convolving = convolv;
				trc->dotType = dot_type; trc->poskern = pos_kern;
				trc->rderivet = r_drive; trc->strideJump = stride_jump;
				trc->kernelSize = kern_size;
				trc->nbyout = outsz;//dual incoder on, �Ŀ� class loss���� �� ����ϹǷ� ����
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
			//�� ��Ʈ���̵��� �����(���� ����)�� ������ Ŀ�ν��ܵ鸸 ��Ƽ� Ű�� �����Ѵ�.
			stepcode = stepcode->slice({ { }, { }, { kernsz - 1 } });//[batch, nt_part, k, zcode, latent][1,4,1,z,32]
			stepcode = stepcode->reshape({ -1, nt_part * zsz, latsz });//[batch, nt_part*k*zcode, latent][1,4*1*z,32]
			//����(�Ϲ���)������ ������ �����(��������)�� Ű�� ���ټ� �����Ͽ� �� ���������� ��ü�� ����� ����� ����Ѵ�.
			t_code = attention_layer2(t_code, stepcode, latsz);//[batch, nt_seq, latent][1,4*8*z,32]
		}
		t_code = stride_fit(t_code, trc->inplus ? seqsz : outsz);//inplus�̸� ���
		//������� ���̰���Ʈ(inp) ���� �������� seqsz�� �ȴ�. outsz�� ��ǥ������ �������̴�.
		//������ ��Ʈ���̵带 Ŀ�λ������ ���������� ��ø�Ǿ� stepcode�� ��Ʈ������ �þ�Ƿ�
		//��� ����� �����.
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
			if(out->fshape[1] > nseq && out->fshape[1] % nseq == 0) {//���߿� conv1d�����Ͽ�
				//���յ��� �ʾƵ� ������ ������ �� �ְ� �Ѵ�.
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
	//eternal code(icode)�߰����
	//1.large demension 1�� ���� icode�� �غ��Ѵ�.
	//2.pcode�� ���̰� 7���� �ǰ� �Ѵ�.(8���̾� ����ȴٸ� pcode�����Է� ������ 448 -> 56 -> 7�μ� �ι� ����)
	//3.pcode�� ������ icode�� ������ ���� �÷� pcode2 icode�� 1�� ���̸� pcode�� ���� 7���� �÷� icode2.
	//4.pocde2�� query, icode2�� key, icode2�� 7���̿� ��� �����(pcode�� �����)���� �ϴ� �÷����� �����Ͽ� value attension���Ѵ�.
	//5.�Ǵ� pcode�� query, 1������ icode�� value���Ͽ� query * value.T�� ����, value�� ���̰� 1���̹Ƿ� ����Ʈ�ƽ����� 
	//	1�����̿� ��� �����(pcode�� �����)���� �ϴ� �÷����� �����Ͽ� [query * value.T]�� ���Ͽ� icode�� query�� pcode����
	//6.icode�� pocde2�� �����ϰ� 5)���� ������ pcode�� ������ڴ��� �����Ͽ� �����ĵ��� ������ �ݿ��� icode�� pocde2�� 
	//	���� ���� ���࿡�� cat�Ͽ� 8���̷� ����� �� �����Ͽ� 1�� ���� icode�����ϰ� ���� icode�� �����־�
	//7. 6)���� ������ icode�� 2)������ ����.
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
		//���ϴ� �����ڵ�(iCode) �н� �� ��ȭ ���Ӽ�(history)�� �����ϴ� �߷� �ڵ�
		ppCode->adjust(pcode);//ù��° ���࿡�� ppCode�� ��ġ�� pcode�� ��ġ�ǰ� Ȯ��.
		auto ei_code = flux(tcrAnt, { -1, 1, fsz }, lat_tAnt, constant);
		ei_code->fill(1.0, pcode);
		ei_code = ei_code->mul(iCode);//[batch, 1, fsz] = [1, fsz], icode ��ġ Ȯ��
		ei_code = concat({ ei_code, ppCode }, 1);//[batch, 1 + npseq, fsz]
		//�� ���� begin, iCode�� �����Ŀ� ���� �ݿ��ȴ�.
		floatt low_bound = trc->lowbound;
		intt convolv = trc->convolving, stride_jump = trc->strideJump, dout = trc->nbyout;
		sytet dot_type = trc->dotType, pos_kern = trc->poskern, pos = trc->positional;
		//��Ʈ���̵� ������ ���� ������ �����Ƿ� t step�� t�ð������� �� ��Ʈ���̵��� ������ t��������
		//generic���� ���� ���� �����ڵ� ����, ��Ʈ���̵尡 8�� �̻��̸� zcode�� 2���̻� ������� �ƴ�
		trc->lowbound = 0.125;//lower bound, ���� ���Ѽ��� 8 Ŀ�λ����� �� 1 �� ����, �̰��� 0.25��
		//�Ͽ� 8 Ŀ�λ����� �� 2�� �Ǿ� ���������� ��»���� 1�εǰ� �ص� ��»���� �ּ� 2���ϰ� �ɼ�����.
		trc->convolving = 8;//������, �Է��� �� ��������� ��¾����ڵ带 1������� ���, �Է±��̰� �� 
		//����� ������ 8Ŀ�δ����� 1�����̾� ����, ��)�Է±��� 8->1, 16->2, �������� 16�̸� 
		//�Է±��� 8->1, 16->1, 32->2, �������� 32�̸� �Է±��� 8->1, 16->1, 32->1, 64->2 
		trc->dotType = ORTHO_DOT;//����(����)�� �ĺ����(CNN)�� ����(����)�� ������.
		trc->poskern = 3;
		trc->positional = 0;
		trc->strideJump = -1;//�ʱ� �Է¸� ������ ��Ʈ���̵�� �ϰ� �Ʒ��������� �⺻ ��Ʈ���̵�(width����)���� �ǵ��� ����.
		trc->nbyout = -1;//dual incoder off
		Generic *icout = generic(ei_code, nullx, fsz, 0, 0, 0, -1, -1, "icode_update");
		ei_code = icout->zcodec;//[batch, 1, fsz], [�����ڵ� + ���������ڵ�]�������� �� ����
		trc->lowbound = low_bound; trc->convolving = convolv;
		trc->dotType = dot_type; trc->poskern = pos_kern;
		trc->positional = pos; trc->strideJump = stride_jump;
		trc->nbyout = dout;
		//�� ���� end, ei_code = [batch, 1, fsz]
		ei_code = ei_code->transpose({ 0, 2, 1 });//[batch, fsz, 1]
		ei_code = ei_code->layer_dense(pcode->fshape[2], actfAnt, Initializer::xavier, "ec_dense");//[batch, fsz, feat]
		auto ep_code = pcode->layer_dense(fsz, actfAnt, Initializer::xavier, "ec_dense2");//[batch, npseq, fsz]
		//auto pp_code = pcode->matmul(ei_code, 2);//[batch, npseq, fsz] = [batch, npseq, feat] * [batch, feat, fsz]
		auto pp_code = *ep_code + *ppCode;//[batch, npseq, fsz], residual(�����Ĵ� ����) & ���Ӽ� ����, ���Ӽ��� ���� �� ���ܰ� �� ���� �� ����
		pp_code = pp_code->actf(actfAnt);
		ovCode = pp_code->overwrite(ppCode);//[batch, npseq, fsz], ppCode������ ��ȭ�� ���Ӽ� �����ȴ�.
		teternal = ep_code->matmul(ei_code);//[batch, npseq, feat] = [batch, npseq, fsz] * [batch, fsz, feat]
		teternal = teternal->actf(actfAnt);
		//���ϴ� ��ȭ ���Ӽ�(history)�� �������� �ʰ� eternal code(icode)�� pcode�� ���� �߷��ϴ� �ڵ�
		auto ei_code2 = flux(tcrAnt, { -1, 1, fsz }, lat_tAnt, constant);
		ei_code2->fill(1.0, pcode);
		ei_code2 = ei_code2->mul(iCode);//[batch, 1, fsz] = [1, fsz], icode ��ġ Ȯ��
		ei_code2 = ei_code2->transpose({ 0, 2, 1 });//[batch, fsz, 1]
		ei_code2 = ei_code2->layer_dense(pcode->fshape[2], actfAnt, Initializer::xavier, "ec_dense3");//[batch, fsz, feat]
		auto ep_code2 = pcode->layer_dense(fsz, actfAnt, Initializer::xavier, "ec_dense4");//[batch, npseq, fsz]
		reternal = ep_code2->matmul(ei_code2);//[batch, npseq, feat] = [batch, npseq, fsz] * [batch, fsz, feat]
		reternal = reternal->actf(actfAnt);

		return teternal->switchout(reternal);
	}
	//pcode - �� ������ڴ����� �������� �������� �ʴ� �Է� ��Ʈ �ڵ�, inp - (�Է�+�߷����), 
	//outsz - �߷���¸��� ���̷μ� inp�� ���������� outsz�� �� ������� �Է���Ʈ�μ� �߷���°� �Բ� 
	//�������� �����ǰ� �� �Է���Ʈ ���̸� strideAnt�� ����� �ǰ��ϰ� �ּ� kernel���� ũ�� �����Ͽ� 
	//time�����Ǵ� kernel ��迡 �ɸ��� �ʰ� �Ѵ�. inp������ ���̰� outsz�� ������ inp�� �߷���¸� 
	//���ԵǴ� ���̰� �Է��� pcode�θ� ���Եȴ�, �߷���� - <s> + Ÿ��
	//chatbot�������� <s> ������ū�� �����Ϳ��� �����ϰ� �����Ѵ�. ������ū�� 0�����μ� 
	//auto regression���� go ��ū���� ���Ǳ� ������
	//indiscret�� �̻굥������ ��� �ǹ��ְ� �̶� inp�� �Է°� Ÿ�� ���� ��� ���� ���� discrete������
	//�̰� �Է���Ʈ(ni_part)�� ���� Ÿ�ٸ� ���� ��� Ÿ�ٸ��� discrete�������̴�.
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
		//�������̸� ������ ���̸� Ŀ�� ������/��Ʈ���̵��� ����� �Ͽ� �ι�° ���̽��� �����Ѵ�.
		//�������� Ŀ���� ���յ��� ������(��Ʈ���̵忡 ���� �ƴҼ��� ������, ���� ��� ������32, Ŀ��8, 
		//��Ʈ���̵�6�̸� ��Ʈ���̵� 4��°(�ɼ� 24)���� ���߾� �����е����� ����) ��������� �����ǹǷ� �н� ȿ���� ��������.
		if(trc->ebatch_t == 2) {
			Flux *a;
			a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			a->fill(1.0);
			expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - ��Ʈ���̵� ���� �Ļ� ������ ����
			//expand_mask->shape();
			//expand_mask->printo(1,2);
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			inp = expand_batch2(inp, expand_mask, strideAnt, kernsz, zpadAnt, noc);//��ġ�� ��� ���������� �Ļ� ���� ������ ����
			//inp->printo(1, 2);
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, strideAnt, r_contig, zpadAnt, 0, 0);//üũ�� �ڵ�
			//if(a->fshape[0] != inp->fshape[1]) throwFault(-1, "check error\n");
		} else {
			auto *a = flux(tcrAnt, { 1, width, featsz }, lat_tAnt, constant);
			a->fill(1.0);
			expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
			n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - ��Ʈ���̵� ���� �Ļ� ������ ����
			//expand_mask->shape();
			//expand_mask->printo();
			//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
			inp = expand_batch(inp, expand_mask, interv, kernsz);//��ġ�� ��� ���������� �Ļ� ���� ������ ����
			//inp->printo(1, 2);
			//a = flux(tcrAnt, { 1, seqsz }, tint, constant);
			//a->arange(seqsz);
			//a = rsc::rsc_combination(a, kernel, stride, r_contig, zpadAnt, 0, 0);//üũ�� �ڵ�
			//if(a->fshape[0] != inp->fshape[1]) throwFault(-1, "check error\n");
		}//inp == [batch, bindbatch(party*derive), kernel, featsz]
		//Ŀ�� ������� expand�� �̿�ȭ�Ͽ� Ȯ���� 2�� 8������ �ϰ� Ŀ�ν����� 16���� 32�� �Ͽ� 
		//��Ʈ���̵� 2�� Ȥ�� 4���� 8Ŀ�ν��� ������ �ǹ̰� ������� ���� ��Ʈ���̵�� ������ ����� �ǹ�
		//�κ��� ���۵ȴ�. ������ �ø��� ���� ������ Ŀ�� ���ܸ� �ø���.
		//residual�� ���� ��� ���� ����� ��ġ��Ų��.
		if(featsz > latsz) latsz = featsz;//���İ��� ���� ū ���� latsz���� ����.
		if(pcode) {
			if(trc->eternity) pcode = eternalCode(pcode);
			if(pcode->fshape[pcode->fdim - 1] > latsz) latsz = pcode->fshape[pcode->fdim - 1];
		}
		if(indiscret > 0) {//�Է� �Ӻ���
			if(inp->fshape[3] * embedim >= latsz) {//Ȥ�� �Է� ���İ� ���������ϰ�� �� ������ �Ӻ����Ͽ�
				latsz = inp->fshape[3] * embedim;//�ϳ��� ��ģ ����� ���������� ũ�� ������ �׻������ �ø���.
			} else {//�ϳ��� ��ģ������(���������� �ƴϸ� 1��������)���� ������ ��ũ�� �Ӻ�������� �÷���
				embedim = latsz / inp->fshape[3];//���� ����� �����.
				latsz = inp->fshape[3] * embedim;
			}
			if(inp->fshape[inp->fdim - 1] == 1) inp = inp->squeeze(inp->fdim - 1);//[batch, bindbatch(party*derive), kernel]
			Flux *expand = nullx;
			if(trc->dualEmbed < 3 && (inp->fshape[1] % n_derive) == 0) {//�Է°� ��� ������ n_derive������ �¾ƶ������
				intx nby = inp->fshape[1] / n_derive;
				intx step = seqsz / nby;
				intx in_derive = (insz / step) * n_derive;//���̰���Ʈ���� �Է� ��Ʈ�� �Ļ� ������ ����
				if(trc->dualEmbed == 0 || trc->dualEmbed == 2) {//�Է� ��Ʈ�� �Ӻ��� Ȥ�� �Է°� ��� ���� �Ӻ���
					auto embed = inp->slice({ {}, {0, in_derive} });
					expand = inp->slice({ {}, {in_derive, -1} });//�����Ʈ�� �ܼ� ���� Ȯ�� Ȥ��
					if(trc->dualEmbed == 2) expand = embedding(expand, outdiscret, embedim, 1);//��� ��Ʈ ���� �Ӻ���
					inp = embed;
				} else {//��� ��Ʈ�� �Ӻ���
					expand = inp->slice({ {}, {0, in_derive} });//�Է���Ʈ�� �ܼ� ���� Ȯ��
					inp = inp->slice({ {}, {in_derive, -1} });
				}
				if(trc->dualEmbed != 2) {
					expand = expand->expand_dims(-1);
					expand = expand->layer_dense(embedim, actfAnt, Initializer::xavier, "inp_embed");
				}
			} 
			inp = embedding(inp, indiscret, embedim, 1);//[batch, bindbatch(party*derive), kernel, featsz(embedim)]
			if(trc->dualEmbed < 3) {
				if(trc->dualEmbed == 0 || trc->dualEmbed == 2) inp = concat({ inp, expand }, 1);//�Է� ��Ʈ �Ӻ����� �����Ʈ �ܼ�Ȯ�� ����
				else inp = concat({ expand, inp }, 1);//�Է���Ʈ �ܼ�Ȯ��� �����Ʈ �Ӻ��� ����
			}
			//�Է� ���İ� ���������̸� �� ���ĸ� �Ӻ����Ŀ� ���ĸ� �Ѱ��� ���Ѵ�.
			if(inp->fdim > 4) inp = inp->reshape({ inp->fshape[0], inp->fshape[1], inp->fshape[2], -1 });
			if(inp->fshape[inp->fdim - 1] != latsz) {
				printf("illegal embedim\n");
				exit(1);
			}
			featsz = latsz;
		} else if(featsz != latsz) {//inp�� Ÿ�ٸ� �ְ� �߷��Ҷ��� �ƹ��͵� ����
			featsz = latsz;//go��ū �ϳ��� �ִ� ���� ��ġ��Ų��.
			inp = inp->layer_dense(featsz, actfAnt, Initializer::xavier, "inp_feat");
		}//inp - [batch, bindbatch(party*derive), kernel, featsz(embedim)]
		intt n_bind = inp->fshape[1];//�Ѱ� �������� �� �Ļ� ���� �Ôн� ����
		intt n_part = n_bind / n_derive;//seqsz / kernsz
		if(trc->positional > 0) {
			trc->positional = 0;//ó�� �ѹ��� �����ų�
			inp = inp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
			inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, derive, party, kernel, featsz]
			auto enpos = flux(trc, 3, &inp->fshape[2], inp->qType, constant);//[party, kernel, featsz]
			enpos->sinpos(n_part * kernsz);//[seq(party, kernel), featsz]
			inp = *inp + *enpos;//sinuosid positional, [batch, derive, party, kernel, featsz] = [batch, derive, party, kernel, featsz] + [positional(party, kernel), featsz]
			inp = inp->transpose({ 0, 2, 1, 3, 4 });//[batch, party, derive, kernel, featsz]
		} else inp = inp->reshape({ -1, n_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
		
		//seqsz(�Է�+Ÿ��)���� �Է���Ʈ ������ strideAnt�� ����� �ǰ��ϰ� �ּ� kernel���� ũ�� �����Ͽ�
		if(seqsz % n_part || (seqsz - outsz) % (seqsz / n_part)) {//time�����Ǵ� kernel��迡 
			throwFault(-1, "dual kernel size not align\n");//�ɸ��� �ʰ� �Ѵ�. ni_part�� 0�̸�
		}									//�Է���Ʈ�� �������� �������� �ʰ� pcode�θ� �����ȴ�.
		intt psz = (pcode ? pcode->fshape[1] : 0);
		intt cross_out = trc->szDualReduce;//Ŀ�� �����쳻 �Ļ� ���� ������ �������� ��ҵǴ� ��� ����
		intt ni_part = (seqsz - outsz) / (seqsz / n_part);//�Է���Ʈ�� ��Ʈ���̵� ����
		if(prtbuild) printf("dual enocder bysz: %d tsz: %d p_sz: %d stride: %d kernel: %d derive: %d ni_part: %d party: %d reducing: %d outsz: %d\n",
			seqsz, outsz, psz, strideAnt, kernsz, n_derive, ni_part, n_part, cross_out, outsz);
		//inp - [batch, party, derive, kernel, featsz]
		Flux *in_part = nullx, *tar_part = nullx;
		if(trc->nblockChain > 1 && trc->inplus == 0 && insz) {//residual�ɼ��ε� inplus�� 0�̸� Ÿ�ٰ�
			//�� ����ϴ� �ɼ��ε�, �̿� ���Ͽ� ���̰���Ʈ�� �Է°��� ������ �Է°� �κи� �����Ͽ� �ؿ��� 
			//�����־��Ҷ� �Է°��� ��ģ��. ��Ʈ���̵带 Ŀ�λ���� ��ø�ǰ� �Ұ�� �̿ɼ����� �ϸ�ʵȴ�.
			//������ Ȯ����Ŀ� �Է���Ʈ�� �����ϴ� �͵� expand�� ������ �Է� ��Ʈ�� Ÿ����Ʈ�� �и��ϴ� �͵�
			in_part = inp->slice({ { }, { 0, ni_part} }); //�Ұ��� �ϹǷ�, ��ø�ҷų� inplus�ɼ����� �����Ѵ�.
			if(trc->resiFirst) tar_part = inp->slice({ { }, { ni_part, -1 } });
		} else if(trc->resiFirst) tar_part = inp;
		if(trc->nblockChain == 0) trc->nblockChain = 1;
		out = inp;
		for(intt i = 0; i < trc->nblockChain; i++) {
			sprintf(s, "dual_chain_%d", i);
			stepcode = dualEncoderLayer(pcode, out, seqsz, outsz, n_bind, n_part, n_derive, 
				kernsz, latsz, featsz, sz_head, ni_part, psz, cross_out, s);
			if(i + 1 < trc->nblockChain) {// || trc->resiopt) {//�������� �ٽ� Ȯ���ϸ� ���°� [batch, seq, feat]
				//�� ���� �����Ƿ� Ÿ�ٰ� ���� �н��� �Ҽ������Ƿ�[ ��)�� �Բ� �����Ͽ� ] �߰��� ����, 
				//���߿� Ȯ���� �����־��� ����� �����ϴ°� ����.
				if(trc->ebatch_t == 2) stepcode = expand_batch2(stepcode, expand_mask, strideAnt, kernsz, zpadAnt, noc);
				else stepcode = expand_batch(stepcode, expand_mask, interv, kernsz);
				stepcode = stepcode->reshape({ -1, n_part - ni_part, n_derive, kernsz, featsz });//[batch, party, derive, kernel, featsz]
			}
			if(trc->resiFirst == 0) {//�Է��� ���� ������ �����־�
				if(i + 1 < trc->nblockChain) {//�߰� ���ڴ� ���̸� ������ ������ �Է���Ʈ�� ������ 
					if(in_part) stepcode = concat({ in_part, stepcode }, 1);//�̸� ��¿� ��ħ.
				}// else {//��.������ ���ڴ� ���̸� ���̰���Ʈ�� in+target�� ��� ���� ������ ��¿��� Ÿ����Ʈ��
				//	//�����Ͽ� �ؿ��� �����־��̸� Ÿ����Ʈ�� ���Ҽ� �ְ� �Ѵ�.
				//	if(trc->resiopt && in_part) out = out->slice({ { }, { ni_part, -1} });
				//}
				if(trc->resiopt && i + 1 < trc->nblockChain) {//�߰����ڴ��̸� �Է�+Ÿ�� ��ü�� �����Ͱ� 
					out = *out + *stepcode;//���ϴ� ���̰� ������ ���ڴ��̸� Ÿ�ٸ� ����
					if(trc->dual_lnorm) {
						sprintf(s, "dual_lnorm_%d", i);
						out = out->layer_normal(s);
					}
					out = out->actf(actfAnt);
				} else out = stepcode;
			} else {//�����־��� �Է� ��
				if(i + 1 < trc->nblockChain) {//�߰� ���ڴ� ���̸� ������ ������ �Է���Ʈ�� ������
					if(trc->resiopt) stepcode = *tar_part + *stepcode;
					if(in_part) out = concat({ in_part, stepcode }, 1);//�̸� ��¿� ��ħ.
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
	//pcode - �� ������ڴ����� �������� �������� �ʴ� �Է� ��Ʈ �ڵ�, inp - (�Է�+�߷����), 
	//outsz - �߷���¸��� ���̷μ� inp�� ���������� outsz�� �� ������� �Է���Ʈ�μ� �߷���°� �Բ� 
	//�������� �����ǰ� �� �Է���Ʈ ���̸� strideAnt�� ����� �ǰ��ϰ� �ּ� kernel���� ũ�� �����Ͽ� 
	//time�����Ǵ� kernel ��迡 �ɸ��� �ʰ� �Ѵ�. inp������ ���̰� outsz�� ������ inp�� �߷���¸� 
	//���ԵǴ� ���̰� �Է��� pcode�θ� ���Եȴ�, �߷���� - <s> + Ÿ��
	//chatbot�������� <s> ������ū�� �����Ϳ��� �����ϰ� �����Ѵ�. ������ū�� 0�����μ� 
	//auto regression���� go ��ū���� ���Ǳ� ������
	//indiscret�� �̻굥������ ��� �ǹ��ְ� �̶� inp�� �Է°� Ÿ�� ���� ��� ���� ���� discrete������
	//�̰� �Է���Ʈ(ni_part)�� ���� Ÿ�ٸ� ���� ��� Ÿ�ٸ��� discrete�������̴�.
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
		//inp == [batch, seqsz, featsz], residual�� ���� ��� ���� ����� ��ġ��Ų��.
		if(featsz > latsz) latsz = featsz;//���İ��� ���� ū ���� latsz���� ����.
		if(pcode && pcode->fshape[pcode->fdim - 1] > latsz) latsz = pcode->fshape[pcode->fdim - 1];
		if(indiscret > 0) {//�Է� �Ӻ���, �Է� ���İ� ���������� ��� ����° ������ �� ���Ұ� �Ӻ����ȴ�.
			if(inp->fshape[2] * embedim >= latsz) latsz = inp->fshape[2] * embedim;
			else {
				embedim = latsz / inp->fshape[2];
				latsz = inp->fshape[2] * embedim;
			}
			if(inp->fshape[inp->fdim - 1] == 1) inp = inp->squeeze(inp->fdim - 1);
			inp = embedding(inp, indiscret, embedim, 1);//[batch, seqsz, featsz(embedim)]
			//�Է� ���İ� ���������̸� �� ���ĸ� �Ӻ����Ŀ� ���ĸ� �Ѱ��� ���Ѵ�.
			if(inp->fdim > 3) inp = inp->reshape({ inp->fshape[0], inp->fshape[1], -1 });
			if(inp->fshape[inp->fdim - 1] != latsz) {
				printf("illegal embedim\n");
				exit(1);
			}
			featsz = latsz;
		} else if(featsz != latsz) {//inp�� Ÿ�ٸ� �ְ� �߷��Ҷ��� �ƹ��͵� ����
			featsz = latsz;//go��ū �ϳ��� �ִ� ���� ��ġ��Ų��.
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
		//���̰���Ʈ�� [|input] + <go token> + target�� �����ϰ�~
		if(trc->inplus) {//�Է�(������ ����)�� �����ϴ� ���̰���Ʈ ��ä�� ��ǥ������ �ϴ� ����ε�
			if(seqsz != stepcode->fshape[1]) {//������ �����ڵ尡 ���ؠ����� ���̰���Ʈ �κи� �����Ѵ�.
				stepcode = stepcode->slice({ { }, { stepcode->fshape[1] - seqsz, -1} });
			}//else �����ڵ尡 ���� ���, stepcode�� ���̰���Ʈ�� ���� �������̹Ƿ� �״��
			//~��ǥ������ input + target + <end token>�� �����Ͽ� feed�ؾ� �Ѵ�.
			//���� pre-train�ҰŸ� ���̰���Ʈ�� �����ϰ� ��ǥ���� �����Ѵ�.
		} else if(seqsz != outsz) {//outsz�� ��ǥ�� ������, ���̰���Ʈ�� �Է��� ���Ե��ִ� ���� ��ǥ���� ����
			stepcode = stepcode->slice({ { }, { stepcode->fshape[1] - outsz, -1} });
			//~��ǥ������ target + <end token>�� �����Ͽ� feed�ؾ� �Ѵ�.
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
		if(seqsz <= 4) kernsz = 4;//�������� 4���� ũ�� Ŀ���� �־��� Ŀ�� ������� �ǰ� ��������
									//Ŀ�ο� ���յ��� ������ ���ڸ��� ���� �����е� Ȯ��ȴ�.
		intt nfinal = (seqsz / kernsz >= 4 ? seqsz / kernsz : 0), resiopt = trc->resiopt;
		if(nfinal && (resiopt == 1 || resiopt == 3)) {//�� ��Ʈ ������ Ŀ�ν��ܿ� �ش��ϴ� �Է� ����, input or all residual
			residual = inp->reshape({ -1, nfinal, kernsz, featsz });//[batch, party, kernsz, featsz]
			residual = residual->slice({ { }, { }, { kernsz - 1 } });//[batch, party, kfinal, featsz]
			residual = residual->reshape({ -1, nfinal, featsz });//[batch, party, featsz]
		}
		intt width = (kernsz < widthAnt ? kernsz : widthAnt);
		//�������̸� ������ ���̸� Ŀ�� ������/��Ʈ���̵��� ����� �Ͽ� �ι�° ���̽��� �����Ѵ�. �������� Ŀ���� ���յ��� 
		//������(��Ʈ���̵忡 ���� �ƴҼ��� ������, ���� ��� ������32, Ŀ��8, ��Ʈ���̵�6�̸� ��Ʈ���̵� 4��°(�ɼ� 24)���� 
		//���߾� �����е����� ����) ��������� �����ǹǷ� �н� ȿ���� ��������.
		floatt r_contig = (width < 8 && rExContigAnt < 0.7 ? 0.7 : rExContigAnt);//�����۰� rate�� ������ ���հ����� 0�� �ǹǷ� �̶��� ������ ������.
		Flux *a = flux(tcrAnt, { 1, width, featsz }, dat_t, constant);
		a->fill(1.0);
		auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
		n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - ��Ʈ���̵� ���� �Ļ� ������ ����
		//expand_mask->shape();
		//expand_mask->printo(1,2);
		//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
		dexp = expand_batch2(inp, expand_mask, strideAnt, kernsz, zpadAnt, noc);//��ġ�� ��� ���������� �Ļ� ���� ������ ����

		intt n_bind = dexp->fshape[1];//�Ѱ� �������� �� �Ļ� ���� �Ôн� ����
		intt n_part = n_bind / n_derive;//transform stride������ ������ ���̿� ������� �¾� �������� �ؾ���, �ƴϸ� transform���� zero padding�����ؾ���.
		if(prtbuild) printf("nest chain seq sz: %d feat sz: %d stride: %d kernel: %d derive: %d party: %d\n", seqsz, featsz, strideAnt, kernsz, n_derive, n_part);
		if(tcrAnt->dbgStep == 1) dexp = dexp->bypass("111\n");
		if(trc->positional > 0) {
			if(trc->positional == 1) trc->positional = 0;//ó�� �ѹ��� �����ų�, �����̸� ���� �����־� �� ��� �����ų�
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
		if(nfinal) {//��ȣ�� ����
			auto kfinal = reduct->slice({ { }, { kernsz - 1 } });//[batch, final_k, party, latent, 1]
			kfinal = kfinal->reshape({ -1, n_part, latsz });//[batch, party, latent], final_k�� derive�� 1�̵ǹǷ�
			if(residual) {
				//kfinal = kfinal->layer_normal();
				//residual = residual->layer_normal();
				kfinal = *kfinal + *residual;//input or all residual
				//kfinal = kfinal->layer_normal();
			}
			if(prtbuild) { printf("nest chain call[batch, party, latent]\n"); kfinal->shape(); }
			kfinal = nestChain(kfinal, latsz, -1, "nest_call");//[batch, party, latent]
			kfinal = kfinal->transpose({ 0, 2, 1 });//[batch, latent, party], ��������� ȥ�յ� kfinal������ �������� �Է� ������� Ȯ��.
			stepcode = kfinal->layer_dense(seqsz, actfAnt, Initializer::xavier, "nc2");//[batch, latent, seqsz]
			stepcode = stepcode->transpose({ 0, 2, 1 });//[batch, seqsz, latent]
		} else {//��Ʈ�� 4�� ���� ������ ���̻� ��ȣ��ʰ� ��������� ������ ���� ������ Ȯ��.
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
		//�������̸� ������ ���̸� Ŀ�� ������/��Ʈ���̵��� ����� �Ͽ� �ι�° ���̽��� �����Ѵ�. �������� Ŀ���� ���յ��� 
		//������(��Ʈ���̵忡 ���� �ƴҼ��� ������, ���� ��� ������32, Ŀ��8, ��Ʈ���̵�6�̸� ��Ʈ���̵� 4��°(�ɼ� 24)���� 
		//���߾� �����е����� ����) ��������� �����ǹǷ� �н� ȿ���� ��������.
		floatt r_contig = (width < 8 && rExContigAnt < 0.7 ? 0.7 : rExContigAnt);//�����۰� rate�� ������ ���հ����� 0�� �ǹǷ� �̶��� ������ ������.
		Flux *a = flux(tcrAnt, { 1, width, featsz }, dat_t, constant);
		a->fill(1.0);
		auto expand_mask = trc->tcr_combination(a, width, width, r_contig, zpadAnt, 0);
		n_derive = emaskAlignment(trc, expand_mask, sz_head);//n_derive - ��Ʈ���̵� ���� �Ļ� ������ ����
		//expand_mask->shape();
		//expand_mask->printo(1,2);
		//[batch, bindbatch(party*derive), seq(kernel), featsz] = [batch, seq, featsz]*[derive, kernel, featsz]
		dexp = expand_batch2(inp, expand_mask, strideAnt, kernsz, zpadAnt, noc);//��ġ�� ��� ���������� �Ļ� ���� ������ ����
		intt n_bind = dexp->fshape[1];//�Ѱ� �������� �� �Ļ� ���� �Ôн� ����
		intt n_part = n_bind / n_derive;//transform stride������ ������ ���̿� ������� �¾� �������� �ؾ���, �ƴϸ� transform���� zero padding�����ؾ���.
		if(prtbuild) printf("block chain seq sz: %d feat sz: %d stride: %d kernel: %d derive: %d party: %d\n", seqsz, featsz, strideAnt, kernsz, n_derive, n_part);
		if(trc->positional > 0) {
			if(trc->positional == 1) trc->positional = 0;//ó�� �ѹ��� �����ų�, �����̸� ���� �����־� �� ��� �����ų�
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
	if(n) {//width�� �����κ��� �������� ������ ������ ��Ʈ���̵� ������ ������ ���յ��� ������ �������� ���ڸ��� ������ ����
		Flux *pad = flux(tcrAnt, { ins->fshape[0], n, ins->fshape[2] }, ins->qType, constant);//�е��Ͽ� ���Ѵ�.
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