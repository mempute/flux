
#include "anet.h"

//indiscret - ����� ���� ��� vocabulary size(��ü �н� ���忡�� ����ũ�� ��ü �ܾ� ����), embedim - �ܾ� �Ӻ����� ������
NameScope *Generic::cbuild(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, 
	intt outdiscret, intt embedim, sytet step, sytet af, floatt lr, const bytet *name)
{
	Trace *trc = TRACER(tcr);
	NameScope *nsc = nullx;
	if(name) nsc = trc->namescope(name);
	if(af < 0) af = ACTF_TANH;
	Anet *anet = new(trc)Anet(trc, latent_sz, trc->actfType < 0 ? af : trc->actfType);
	bool dual_code = (TRACER(anet->tcrAnt)->nbyout < 0 ? 0 : 1);
	anet->optAnet();
	canet = anet;
	if(ingate) {
		zcodec = anet->encodeGate(ingate, targate->fshape[1] * -1, indiscret, embedim);
		zcodec2 = anet->encodeOut;//Ȱ���Լ� �� �÷���
		if(targate == nullx) goto LB1;
	}
	//zcodec - C_ZZ�̸� [batch, outsz(n_stride*n_reduce), n_derive], C_XT�̸� //[batch, bindbatch, outsz(cross_out)]
	if(dual_code) {
		if(ingate == nullx) makeAnet(trc, latent_sz, af);
		if(trc->dualChain) dcodec = ((Anet *)canet)->chainEncoder(zcodec, trc->bygate, trc->nbyout, indiscret, embedim, -1, latent_sz);
		else dcodec = ((Anet *)canet)->dualEncoder(zcodec, trc->bygate, trc->nbyout, indiscret, outdiscret, embedim, -1, -1, latent_sz);
	} else if(targate) dcodec = anet->decodeGate(zcodec, -1);//C_XT ���̽�(������ ���� ���� �ٷ� �ѹ��� Ÿ�� ����)�̹Ƿ� ��� ������ ���� �� -1
	//dcodec - C_ZZ�̸� [batch, outsz(n_stride*n_reduceȮ��), n_derive(hidden)], C_XT�̸� [batch, outsz(cross_out), bindbatch(hidden)]
	ctarget = targate;
	if(trc->outback) {//���� �����ε� ������ڴ� ���࿡�� ����Ǵ� ���̸� ������ڴ� ����ɶ��� �ƿ��� ����
		//�Ǵ� ������ڴ��� �ƴ� �Ϲ� ���ڵ��϶� ����Ǵ� �ɼ��̸� ������ڴ� ������� ������ �ƿ��� ����
		if((trc->dualStep && trc->nbyout >= 0) || (trc->dualStep == 0 && trc->nbyout < 0)) {
			if(name) trc->endscope();
			return nsc;
		}
	}
	cypred = anet->outGate(dcodec, outdiscret, targate->fshape[2], clogit, "conet_out", 0);
	//cypred - discrete�̸� [batch, outsz, 1], �ƴϸ� [batch, outsz, out_feat]
	//clogit - discrete�϶� [batch, out_seq, vocab_sz], �ƴϸ� cypred�� ����
	if(step == 2) {
		if(name) trc->endscope();
		return nsc;
	}
	closs = anet->calcLoss(clogit, targate);
	if(step == 3) {
		if(name) trc->endscope();
		return nsc;
	}
	coptrain = adam_optimizer(trc, lr)->minimize(closs);
	//coptrain = gradient_descent_optimizer(trc, lr)->minimize(tloss);
LB1:;
	if(name) trc->endscope();
	return nsc;
}
intt Generic::cbuild(Tracer *tcr, Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim, 
	sytet af, floatt lr, const bytet *name, bool auto_encoder)
{//���� ��ҵ� ��ä�ڵ� ���� Ȥ�� �Է��� Ÿ������ �н��ϴ� �������ڴ� ��.
	if(name) tcr->namescope(name);
	makeAnet(tcr, latent_sz, af);
	Anet *anet = (Anet *)canet;
	zcodec = anet->encodeGate(ingate, outsz, indiscret, embedim);
	zcodec2 = anet->encodeOut;//Ȱ���Լ� �� �÷���
	if(auto_encoder == 0) dcodec = nullx;
	else {//auto encoder
		dcodec = anet->decodeGate(zcodec, ingate->fshape[1]);
		cypred = anet->outGate(dcodec, indiscret, ingate->fshape[2], clogit);
		closs = anet->calcLoss(clogit, ingate);
		//caccuracy = anet->calcAccuracy(cypred, ingate);
		coptrain = adam_optimizer(tcr, lr)->minimize(closs);
	}
	dcodec = zcodec;
	if(name) tcr->endscope();
	return zcodec->fshape[1];//tcr ����� ���� ������� ������� ����
}
void Generic::makeAnet(Tracer *tcr, intt latent_sz, sytet af)
{
	if(af < 0) af = ACTF_TANH;
	Anet *anet = new(tcr)Anet(tcr, latent_sz, TRACER(tcr)->actfType < 0 ? af : TRACER(tcr)->actfType);
	anet->optAnet();
	canet = anet;
	zcodec = nullx;
}
void Generic::setmhead(Generic *src)
{
	((Anet *)canet)->szmhead = ((Anet *)src->canet)->szmhead;
}
NameScope *Generic::_coaxial(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet step, sytet af, floatt lr, const bytet *name)
{
	Trace *trc = TRACER(tcr);

	coaxNsp = nullx;
	canet = nullx;
	byPolar = 1;//���ڴ��� ���ڴ� �ΰ� ��� �����ȴٴ� ǥ��, �̰��� �ƴϰ� ���������� ȣ��Ǿ� ���ڴ��� ������ completeConvolve
				//�϶� endecode�ɼ� �������־ ������������ ���� ���ڵ�(����)�ϰ� �Ѵ�.
	if(name) coaxNsp = trc->namescope(name);
	if(ingate && embedim >= 0) compress(tcr, ingate, latent_sz, indiscret, embedim, af, lr);
	if(trc->nbyout >= 0) {//ingate�� �־������� �̴� pcode(���� �����ڵ�)�̴�. �Է��� by����Ʈ��
		bool prev_code = embedim < 0 ? 1 : 0;//���̳����� csnr)���� -1�� �־���� ���̳������� �ڵ����ƴٴ� 
		if(trc->sembedn >= 0) embedim = trc->sembedn;//�����μ� �̸� ���� �ڵ�� ����, dncs)�� ���� stracode��������
		if(trc->slatent >= 0) latent_sz = trc->slatent; //���̳����� �� ������������ �ι� ���ڵ��� ��� ���ڴ��� �Էµȴ�.
		if(canet == nullx) makeAnet(tcr, latent_sz, af);//�ΰų� �ƴ� ����(�̰�� by����Ʈ ��ü�� Ÿ��(�߷�)������)
		if(trc->dualChain) dcodec = ((Anet *)canet)->chainEncoder(prev_code ? ingate : zcodec, trc->bygate,
			trc->nbyout, indiscret, embedim, -1, latent_sz);
		else dcodec = ((Anet *)canet)->dualEncoder(prev_code ? ingate : zcodec, trc->bygate,
			trc->nbyout, indiscret, outdiscret, embedim, -1, -1, latent_sz);//embedim < 0�� ingate��
		//dynagen���� ����� ���� ���� �����ڵ��̰� (������ derivefeat�� ������ ������ڴ��� �����ϰ�
		//�����Ͼ� �Ѵ�. �ȱ׷��� feature������ ���� �ʰ� �ȴ�.->������ڴ����� �����ϹǷ� �������.)
		//embedim�� 0�̻����� ȣ������� zcodec���� ������ �̴� ���ʸ������� ������� ����� ���� �����ڵ�
	} else {//chatbot�� ��� ���ڴ�(auto regression)�� �ƴ� auto encoding���� �����Ҹ� <s> ������ū
		//�� �����Ϳ��� �����ϰ� �����ϸ� 32���̸� 16���̷� �ѹ��� ������ ���ڵ� ���� �Ұ�� 9�� ��������
		//�ɸ��� ������ū�� �־��� ��� 3�� ���������� �н��� �ߵȴ�. ���߿� ����.
		if(targate) decompress(tcr, targate->fshape[1]);
	}
	ctarget = targate;
	if(trc->outback) {//���� �����ε� ������ڴ� ���࿡�� ����Ǵ� ���̸� ������ڴ� ����ɶ��� �ƿ��� ����
		//�Ǵ� ������ڴ��� �ƴ� �Ϲ� ���ڵ��϶� ����Ǵ� �ɼ��̸� ������ڴ� ������� ������ �ƿ��� ����
		if((trc->dualStep && trc->nbyout >= 0) || (trc->dualStep == 0 && trc->nbyout < 0)) {
			if(name && step >= 0) trc->endscope();
			return coaxNsp;
		}
	}
	if(targate) connect(targate, outdiscret, step, trc->learningRate > 0 ? trc->learningRate : lr, trc->optType);
	if(name && step >= 0) trc->endscope();//step �� -1�̸� �� �Լ� ���ο��� ������ Ŭ����
	return coaxNsp;
}
NameScope *Generic::generic(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet step, sytet af, floatt lr, const bytet *name)
{
	//try {
		if(TRACER(tcr)->convolving > 1) return _coaxial(tcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, step, af, lr, name);
		else return cbuild(tcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, step, af, lr, name);
	//} catch(FaultObj e) {
	//}
}

Generic::Generic(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet af, floatt lr, const bytet *name)
{
	generic(tcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, 0, af, lr, name);
}
Generic::Generic(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet af, floatt lr, const bytet *name)
{
	generic(ingate->fxTcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, 0, af, lr, name);
}
Generic::Generic(Tracer *tcr, Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim, sytet af, floatt lr, const bytet *name, bool auto_encoder)
{
	cbuild(tcr, ingate, outsz, latent_sz, indiscret, embedim, af, lr, name);
}
Generic::Generic(Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim, sytet af, floatt lr, const bytet *name, bool auto_encoder)
{
	cbuild(ingate->fxTcr, ingate, outsz, latent_sz, indiscret, embedim, af, lr, name);
}

void Generic::accuracy(Flux *&predicts, Flux *&targets, intt discrete_out) //��Ȯ�� ���� �׶��� ����
{//�н��Ҷ��� �Է°� Ÿ�� ����Ʈ�� ������� �ʰ� ���� ������� �δ� ���� ���� �ڵ� ��Ȯ���� ����Ҷ��� �н� ����¶��� �����Ϳ�
	caccuracy = ((Anet *)canet)->calcAccuracy(predicts, targets, discrete_out);//�����ڵ��� ���°� Ʋ���� �����̴�.
}
Flux *Generic::measureAccuracy(void) //��Ȯ�� ���� �׶��� ����.
{
	((Anet *)canet)->tcrAnt->run({ caccuracy });
	return caccuracy;
}
intt Generic::reduction(intt outsz, intt szlat)
{
	if(TRACER(zcodec->fxTcr)->layer_norm == 1) {
		zcodec = zcodec->layer_normal();
		zcodec = zcodec->actf(ACTF_TANH);
	}//zcodec�� zcodec2�� ���� �������� �����ڵ尡 �����Ѵ�.
	dcodec = zcodec = ((Anet *)canet)->encodeGate(zcodec, outsz, -1, -1, -1, -1, szlat);//�ٸ� ������ ���� ���� ����� ����Ҹ�� zcodec�� �����ϸ� �ȴ�.
	//zcodec = zcodec->bypass("aaa\n");
	//dcodec = zcodec;
	zcodec2 = ((Anet *)canet)->encodeOut;//Ȱ���Լ� �� �÷���
	return zcodec->fshape[1];//tcr ����� ���� ������� ������� ����
}
void Generic::decompose(intt outsz)
{
	dcodec = ((Anet *)canet)->decodeGate(dcodec ? dcodec : zcodec, outsz);
}
intt Generic::decompose2(intt outsz, bool im)
{
	if(TRACER(dcodec->fxTcr)->layer_norm == 1) {
		dcodec = dcodec->layer_normal();
		dcodec = dcodec->actf(ACTF_TANH);
		if(TRACER(zcodec->fxTcr)->hidPrint == 2) zcodec = zcodec->bypass("===========2=============");
	}
	dcodec = ((Anet *)canet)->decodeGate2(dcodec, outsz, -1, -1);
	if(im) zcodec = dcodec;//���� ���� en-decode�̸� �����ڵ� ����.
	return dcodec->fshape[1];//tcr ����� ���� ������� ������� ����
}
//v - Ŀ�� ������, last - Ÿ�� ������ ����, conv - ������ ���
intt first_convolv(intt v, intt last, intt conv)
{
	for(;v * conv < last; v *= conv);
	if(v == last) return conv;//������ ����� ���յǸ� ������ ����� �״�� ����

	return (last / v < 2 ? conv : last / v);//������ ���� Ÿ���� �� 2 �̻��̸� �� ����� ù��° ������ ����� ����
}
intt Generic::reduceKernel(Tracer *tcr, intt in_sz)
{//Ŀ�λ����� ���� �Է� ����� ������ Ŀ�� ����� �� ���� �����Ѵ�.
	Trace *trc = TRACER(tcr);
	intt ksz;

	for(ksz = trc->kernelSize; in_sz < ksz; ksz /= 2) {
		if(ksz / 2 == in_sz) {
			ksz /= 2;
			break;
		} else if(ksz / 2 < in_sz) break;//Ŀ�� ����� 2�� ����� outer�� �����Ѵ�.
	}
	return ksz;
}
NameScope *Generic::compress(Tracer *tcr, Flux *ingate, intt latent_sz, intt indiscret, intt embedim, sytet af, floatt lr, const bytet *name)
{
	Trace *trc = TRACER(tcr);
	intt sz_lowbound, convolv = (trc->convolving > 1 ? trc->convolving * (trc->kernelSize / trc->bandwidth) : trc->convolving);
	NameScope *nsc = nullx;
	//(low bound, convolve, input seq)->(sz_lowbound,compress out sz,,)
	//������ sz_lowbound�� reduction�� �Է»���� 8���� ���Ե����ʰ� �Ѵ�. ���ڸ��� �����е���
	//��) (1,4,32)->(8,8), (0.25,4,32)->(2, 8, 2), (0.25,2,32)->(2,16,8,4,2), (0.25,16,32)->(2,2)
	//(1,4,64)->(8,16,8), (0.5,4,64)->(4,16,4), (1,4,16)->(8,8), (1/8,16,16)->(1,1), (0.5,8,16)->(2,2)
	in_sz = ingate->fshape[1];
	if(name) nsc = tcr->namescope(name);
	
	sz_lowbound = reduceKernel(tcr, in_sz);
	intt save_size = trc->kernelSize;
	trc->kernelSize = sz_lowbound;

	if(trc->bandwidth < sz_lowbound) sz_lowbound = trc->bandwidth;

	sz_lowbound *= trc->lowbound;//������ 8 ���� ����� ���
	if(sz_lowbound == 0) sz_lowbound = 1;
	else if(trc->compbot > 0) sz_lowbound = trc->compbot;
	ist = 0;
	seq_st[ist++] = in_sz;
	if(convolv > 1) in_sz /= convolv;//convolv == 2) in_sz == 16 | convolv == 4) in_sz == 8
	else in_sz /= sz_lowbound;
	if(in_sz < sz_lowbound) in_sz = sz_lowbound;
	//derivefeat == 2 �̸� ingate�� �������� 8 ~ 32���� �̸� 32�� Ȯ��Ǿ� ������ 4�� ���ϵȴ�.
	in_sz = cbuild(tcr, ingate, in_sz, latent_sz, indiscret, embedim, af, lr);//10 * 32 (vocab size)
	if(convolv <= 1) {
		if(name) tcr->endscope();
		trc->kernelSize = save_size;//����
		return nsc;
	}
	if(in_sz > sz_lowbound) {// && in_sz >= trc->bandwidth) {
		while(1) {
			((Anet *)canet)->kernszAnt = reduceKernel(tcr, in_sz);//in_sz�� Ŀ�� ������� �����Ǹ�
			seq_st[ist++] = in_sz;	//Ŀ�λ���� ��� �����ǰ�, �̹� ������������� ���ǹǷ� �����ʿ����.
			in_sz /= convolv;
			if(in_sz < sz_lowbound) in_sz = sz_lowbound;
			//if(trc->ln1) zcodec = zcodec->layer_normal();
			tcr->namescope(ist);
			in_sz = reduction(in_sz);//convolv == 2) in_sz == 16, 8 | convolv == 4) in_sz == 8
			tcr->endscope();
			if(in_sz <= sz_lowbound) break;//==���ǿ��� <= �������� ����, ���߿� ����
		}
	}
	trc->kernelSize = save_size;//����
	//lowbound�� 1 �̻��̸� Ŀ�� ������ ������ �������� ���ϴ� �ɼ��̹Ƿ� completeConvolve(������»���� �ο�ٿ��������
	//���Ե����� �̸� �ѹ��� ȥ���ϴ� �ɼ�)�� �ǹ̾����Ƿ� ���ؾ��Ѵ�.
	//byPolar == 0�̸� ���ڵ��� ������ ��Ʈ��ũ, 1�̸� ���ڵ��� ���ڵ��� ��� ������ ��Ʈ��ũ�μ� completeConvolve�� 1�̸�
	//���ڵ��� ������ ��Ʈ��ũ�� ���� �����ϰ� 2�̸� ���ڵ��� ���ڵ��� ��� ������ ��Ʈ��ũ���� ���ڵ��Ҷ� ���� �����Ѵ�.
	//nxcicp.decode gate2�̸� ���ڵ� �������� ������ ���ڵ��ǹǷ� ���� ���ڵ����� �������������ʰ� ���ڵ��������� ��������ǰ� �Ѵ�.
	if(trc->endecode == 0 && trc->lowbound <= 1 && ((byPolar == 0 && trc->completeConvolve) || (byPolar && trc->completeConvolve == 2))) {//in_sz == 8,  
		//if(trc->ln2) zcodec = zcodec->layer_normal();
		tcr->namescope("end");
		//spotOpt == 3���μ� ���������϶��� ����Ǵ� �ɼ��϶� attsign == 1�̸� ���ڵ��� ������ ��Ʈ��ũ���� ���� ���࿡�� ���ټ� 
		//�����ϰ� ���ڵ��� ���ڵ��� ��� ������Ʈ��ũ�� ���� ���࿡�� ���ټ��� ���� ���ϰ��Ѵ�. spotOpt�� 3�� �ƴϸ� attsign == 1�� �ǹ̾���.
		in_sz = reduction(in_sz, trc->attsign == 1 && byPolar ? (trc->spotlat > 0 ? trc->spotlat * -1 : 0) : trc->spotlat);//residual reduction����
		tcr->endscope();
		((Anet *)canet)->finalReduced = 1;
	}
	if(trc->hidPrint == 1) dcodec = dcodec->bypass("============1============");
	if(name) tcr->endscope();
	save_ist = ist;
	save_insz = in_sz;
	return nsc;
}
void Generic::setFrontendFactor(Generic *from, Flux *_zcodec)
{
	((Anet *)canet)->lat_tAnt = ((Anet *)from->canet)->lat_tAnt;
	((Anet *)canet)->cmodAnt = ((Anet *)from->canet)->cmodAnt;
	((Anet *)canet)->finalReduced = ((Anet *)from->canet)->finalReduced;
	dcodec = zcodec = _zcodec;
	in_sz = from->save_insz;
	ist = from->save_ist;
	//for(intt i = 0;i < ist; i++) seq_st[i] = from->seq_st[i];//����� �� ������ �ʿ������ ���Ѵ�.
}
void Generic::setFrontendFactor2(Flux *in)
{
	((Anet *)canet)->lat_tAnt = in->qType;
	((Anet *)canet)->cmodAnt = C_ZZ;
	dcodec = zcodec = in;
}
NameScope *Generic::decompress(Tracer *tcr, intt tar_sz, sytet _endecode, const bytet *name)
{
	Trace *trc = TRACER(tcr);
	NameScope *nsc = nullx;
	intt convolv = (trc->convolving > 1 ? trc->convolving * (trc->kernelSize / trc->bandwidth) : trc->convolving);

	if(_endecode < 0) _endecode = trc->endecode;
	if(name) nsc = tcr->namescope(name);
	if(tar_sz > in_sz) {
		if(_endecode) {
			//bool ln = trc->ln2; 
			intt cv = first_convolv(in_sz, tar_sz, convolv);//convolv�� ���յ��� ������ ù��°�� �ּ� ���ռ��� ���Ѵ�.
			for(;in_sz != tar_sz;) {
				if(tar_sz > trc->bandwidth && in_sz * cv > tar_sz) {//Ÿ���� Ŀ�� ������� ������ decompose2���� 
					break;//Ÿ�ٻ���� �������Ƿ� �� ���� �����ϰ� Ÿ�� ����� ���յ��� ������ ���ڸ��� ������� �����ΰ��� �����.
				}
				in_sz = (in_sz * cv > tar_sz ? tar_sz : in_sz * cv);
				//if(ln) {
				//	dcodec = dcodec->layer_normal();
				//	if(trc->ln3 == 0) ln = 0;
				//}
				tcr->namescope(ist--);
				in_sz = decompose2(in_sz, 0);
				tcr->endscope();
				cv = convolv;//�� ������ ����
			}
		} else {
			for(--ist;ist >= 0 && in_sz != tar_sz; ist--) {
				if(seq_st[ist] < tar_sz) in_sz = seq_st[ist];
				else in_sz = tar_sz;
				tcr->namescope(ist);
				decompose(in_sz);//convolv == 2) in_sz == 16, 32 | convolv == 4) in_sz == 32
				tcr->endscope();
			}
		}
	}
	if(in_sz != tar_sz) {
		tcr->namescope("end");
		decompose(tar_sz);//�� ���࿡�� in_sz�� tar_sz���� ũ�ԵǴ� ���� ����.�Է��� in_sz�� �� Ŀ�� �� �������
		tcr->endscope();	//������� ���� ���� in_sz�� �� ũ�Եǰ� ����ƴ� ���� �������� ȣ�⿡�� Ÿ�ٻ���� ��������.
	}
	if(name) tcr->endscope();
	return nsc;
}
Flux *Generic::cx_optimizer(Flux *loss, floatt lr, sytet opt_t, vector<Flux *> *weight_list)
{
	Anet *anet = (Anet *)canet;
	Flux *ctrain = nullx;

	switch(opt_t) {
	case 0:
		ctrain = adam_optimizer(anet->tcrAnt, lr)->minimize(loss, weight_list);
		printf("adam opt\n");
		break;
	case 1:
		ctrain = gradient_descent_optimizer(anet->tcrAnt, lr)->minimize(loss, weight_list);
		printf("sgd opt\n");
		break;
	default:
		throwFault(-1, "non def optimizer\n");
	}
	return ctrain;
}
void Generic::connect(Flux *targate, intt outdiscret, sytet step, floatt lr, sytet opt_t)
{
	Anet *anet = (Anet *)canet;
	bool dual_code = (TRACER(anet->tcrAnt)->nbyout < 0 ? 0 : 1);

	cypred = anet->outGate(dcodec, outdiscret, targate->fshape[2], clogit, "connect", 0);
	if(step == 2) return;
	closs = anet->calcLoss(clogit, targate);//�Ϲ��� �߷��ڵ�(<s> + Ÿ��) Ÿ�� ����, targate - Ÿ�� + <e>
	bloss = anet->batchLoss;
	//caccuracy = anet->calcAccuracy(cypred, targate);
	if(step == 1) return;

	if(step < 0) {
		anet->tcrAnt->endscope();//�� ���� ����ġ ������ �н���Ű������ ���ӽ������� �ݴ´�.
		coptrain = cx_optimizer(closs, lr, opt_t, anet->tcrAnt->trainvar(coaxNsp));
	} else coptrain = cx_optimizer(closs, lr, opt_t);
}
Flux *Generic::train(intt *n_train)
{
	Anet *anet = (Anet *)canet;
	Trace *trc = TRACER(anet->tcrAnt);

	if(trc->eternity) {//�����ڵ� �н� ����
		anet->teternal->reentrance(0);//�����ڵ� �ݿ��� �ڵ�����ϰ� �Ѵ�.
		anet->reternal->reentrance(1);//�����ڵ带 ȸ���� �ڵ� ���� ���ϰ��Ѵ�.
		trc->run({ coptrain, closs, anet->ovCode });//ovCode�� ppcode����
	} else trc->run({ coptrain, closs });
	
	return closs;
}
Flux *Generic::_predict(Flux *robjec, Flux **loss_fx) //Ÿ���� �н��Ҷ��� Ʋ���� ����ġ �ε带 �ϹǷ� �н��� �̾ �ٷ� ������ ���
{									//�̹��� �н��� ���� ������ ����ġ ������ �� ���� �ؾ��Ѵ�.->loadedWeightó���� ������.
	Anet *anet = (Anet *)canet;
	Trace *tcr = TRACER(anet->tcrAnt);

	if(tcr->eternity == 1) {//�����ڵ� �����̰� �߷ж��� �����ڵ� �ݿ��� ���ϴ� ���
		anet->teternal->reentrance(1);//�����ڵ� �ݿ��� �ڵ���� ���ϰ��Ѵ�.
		anet->reternal->reentrance(0);//�����ڵ带 ȸ���� �ڵ常 �����ϰ� �Ѵ�.
	} else if(tcr->eternity == 2) {//�����ڵ� �����̰� �߷ж� �����ڵ� �ݿ��ϴ� ���
		anet->teternal->reentrance(0);//�����ڵ� �ݿ��� �ڵ�����ϰ� �Ѵ�.
		anet->reternal->reentrance(1);//�����ڵ带 ȸ���� �ڵ� ���� ���ϰ��Ѵ�.
		if(loss_fx) {
			if(bloss) {
				tcr->run({ robjec, bloss, anet->ovCode });//ovCode�� ppcode����
				*loss_fx = bloss;
			} else {
				tcr->run({ robjec, closs, anet->ovCode });
				*loss_fx = closs;
			}
		} else tcr->run({ robjec, anet->ovCode });
		return robjec;
	}
	//((Anet *)canet)->resizeFhs();
	if(loss_fx) {
		if(bloss) {
			tcr->run({ robjec, bloss });
			*loss_fx = bloss;
		} else {
			tcr->run({ robjec, closs });
			*loss_fx = closs;
		}
	} else tcr->run({ robjec });

	return robjec;
}
//��ġ�� �����ҰŸ� ��ġ����� batchPart���� ���� �Ͽ� ��ġ���� �л� ����ʵ��� �Ѵ�. 
Flux *Generic::inference(Flux **loss_fx)
{
	Trace *tcr = TRACER(((Anet *)canet)->tcrAnt);
	Flux *bgate = tcr->bygate, *go = tcr->gotok, *end = tcr->endtok;
	intt nxin = bgate->fshape[1] - tcr->nbyout;//�Է� ����, ������ 0�̰�, ó������ �߷��ϰ� �Է��� pcode�̴�.
	intt width = go->fshape[1];//�߷� ���� ��ū ����, �� ���� ������ �߷�
	intt rsz = 0;
	Flux *pred;
	void *p;
	//bgate - ���ʸ� �����Ҷ� �Է��� �����ʰ� ���̰���Ʈ �Է°� <s>+Ÿ���� ������ �Էº��� ������ڴ��� 
	//		�н��� ��� �Է°� <s>+Ÿ�� ���̰� ���ʸ��� �Է��� �־��� ��� <s>+Ÿ�ٸ��̸� �̶� �Է��� 
	//		����Ǿ� ������ڴ��� pcode�� ���Եȴ�. �н��� �Է��� �־��� ��� �Է� ��Ʈ�� ������ ���¿��� 
	//		ȣ���ؾ� �Ѵ�.	 bygate�� ���� �����ϹǷ� ���� feed�Լ� ȣ��ȴ´�. ���� feed�� �������� 
	//		�����Ƿ� ��ġ�� �����ص� ��ġ���� �л� ����ʵ��� ū���� �����Ѵ�.
	if(nxin < 0) throwFault(-1, "not auto regression\n");
	if(bgate->fshape[0] != go->fshape[0]) go->resizing2(bgate, "inference");
	if(bgate->fshape[0] != end->fshape[0]) end->resizing2(bgate, "inference");
	for(intt i = 0; i < bgate->fshape[0]; i++) {//���� ��ū �ʱ� ����, netshell, dual chatbot
		for(intt k = 0; k < width; k++) {		//�ҽ������� ���� �ܺο��� �����ϹǷ� �ʿ����.
			p = go->read_p({ i, k }, &rsz);		//���߿� ����
			bgate->write_p({ i, nxin + k }, p, rsz);
		}
	}
/*//���� �߷��� �������� �ʰ� seq to seq �۵� �׽�Ʈ
bgate->printo(2, 3);
Flux *dd = flux(tcr, { bgate->fshape[0], 15, 1 }, tfloat, variable);//�Ʒ� 32���� �۰��ϸ� 
dd->arange(-1);								//end token������ ����Ǿ� ��ġ 1���� ���� �׽�Ʈ
pred = flux(tcr, { bgate->fshape[0], 32, 1 }, tfloat, variable);
pred->fill(0.0);//reset
pred->howrite(dd);//target
end = flux(tcr, { bgate->fshape[0], 1, 1 }, tfloat, variable);//token
end->fill(888.0);
pred->howrite(end, 15);//input*/
	intt j = nxin;//inplus�̸� �Է� + ���� ���� ��µǹǷ� ���������� �����ϱ����� ���۰��� nxin���� ������ 0�� �����Ѵ�.
	if(tcr->inplus) nxin = 0;//inplus�� �н��� �����Ͽ� ���� �����ϰ� �߷ж��� inplus�������� �𵨺����� �߷��ص� �ȴ�.
	for(;;) {//�Է°� Ÿ���� ������ �Էº��� ������ڴ��� �н��� ��� �Է±��̸� ������ Ÿ�ٺ��� �߷�
		pred = _predict(cypred, loss_fx);//�׳� �ܼ��� ���ڸ� ������ �ν��� ���ϵȴ�.
		if(bgate->fshape[0] == 1 && !memcmp(pred->read_p({ 0, j - nxin }), end->begin_p(), end->sizefx())) {
			break;//��ġ�� 1���� ��� ���� ��ū�� �߷еǸ� ����
		}
		j += width;
		if(j >= bgate->fshape[1]) break;//��� ��ġ�� ������ ������ �߷е����� ����
		for(intt i = 0; i < pred->fshape[0]; i++) {//�̹� ������ ��°�� �߷��� ��� �������� �����Ͽ� ���� ������ �Է����� ����
			for(intt k = 0; k < width; k++) {
				p = pred->read_p({ i, j - width - nxin + k }, &rsz);
				bgate->write_p({ i, j + k }, p, rsz);//���̰���Ʈ�� �������� �����ϸ� ����
			}
		}
//bgate->printo(2, 3);
	}
	return pred;
}
Flux *Generic::predict(Flux **loss_fx)
{
	Trace *tcr = TRACER(((Anet *)canet)->tcrAnt);
	if(tcr->predlog) return _predict(clogit, loss_fx);
	else if(tcr->nbyout < 0 || tcr->autoRegress == 0) return _predict(cypred, loss_fx);
	else return inference(loss_fx);
}
/*void Generic::autoEncode(Flux *ingate, intt z_size, intt latent_sz, floatt lr = -1)
{
	Anet *anet = (Anet *)canet;
	zcodec = anet->encodeGate(ingate, z_size, -1, -1);
	dcodec = anet->decodeGate(zcodec, ingate->fshape[1]);
	cypred = anet->outGate(dcodec, ingate->fshape[2], clogit);
	closs = anet->calcLoss(clogit, ingate);
	//caccuracy = anet->calcAccuracy(cypred, ingate);
	coptrain = adam_optimizer(ingate->fxTcr, lr)->minimize(closs);
}*/
NameScope *Algol::algol(Tracer *trc, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, bool contraction, sytet af, floatt lr, const bytet *name)
{	
	Trace *tcr = TRACER(trc);
	NameScope *nsc = nullx, *g_nsc;
	if(name) nsc = tcr->namescope(name);

	Generic *gnet = agnet = new(tcr)Generic;
	Generic *dnet = agnet2 = new(tcr)Generic;
	agtrc = tcr;
	
	if(contraction) {//�Է°� ����� ���� ��Ī, �Է��� ������ ���ڵ� ���
		g_nsc = gnet->generic(tcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, 1, af, lr, "generator");
	} else {//�Է��� �����ڵ�, �ٷ� ���ڵ� ���
		g_nsc = tcr->namescope("generator");
		gnet->makeAnet(tcr, latent_sz, af);
		gnet->setFrontendFactor2(ingate);
		gnet->decompose(targate->fshape[1]);
		gnet->connect(targate, outdiscret, 1, tcr->learningRate > 0 ? tcr->learningRate : lr, tcr->optType);
		tcr->endscope();
	}
	Flux *gen_out;
	if(outdiscret) {//generator�� �̻��� ��� �̺е��� �ʴ� argmax���
		gen_out = gnet->clogit->softmax();//�̺а����� ����Ʈ�ƽ��� ��ü
	} else gen_out = gnet->cypred;
	NameScope *d_nsc = tcr->namescope("discriminator", 1);//�Ʒ� �ι�° ������������ ù��° ����������
	//����ġ�� �����ϱ����� reuse 1�� ����. �ǹ������� �������� �������� Ÿ�ٰ��̰� out discret�� Ÿ���� 
	//�������̰� Ÿ�� �Է��� �����ϴ� ���̹Ƿ� out discret�� �Է°��� ���� �����Ѵ�.
	//gen_out - [batch, target_seq, target_feat]
	dnet->compress(tcr, gen_out, latent_sz, 0, 0, af, lr);
	Flux *d_fake = dnet->zcodec->reshape({ -1, dnet->zcodec->fshape[1] * dnet->zcodec->fshape[2] })->layer_dense(1, "sigmoid", Initializer::xavier, "d_dense");
	if(outdiscret) {
		targate = targate->squeeze(2);//���� 1�� ������ ���ش�.
		targate = targate->one_hot(outdiscret);//[batch, target_seq, target_feat(vocab_sz)]
	}
	dnet->compress(tcr, targate, latent_sz, 0, 0, af, lr);
	Flux *d_real = dnet->zcodec->reshape({ -1, dnet->zcodec->fshape[1] * dnet->zcodec->fshape[2] })->layer_dense(1, "sigmoid", Initializer::xavier, "d_dense");
	tcr->endscope();
	//d_real = d_real->bypass("real part\n");
	//d_fake = d_fake->bypass("fake part\n");
	/*Flux *aa = d_real->log();
	Flux *bb = 1.0 - *d_fake;
	bb = bb->bypass("11\n");
	bb = bb->log();
	aa = *aa + *bb;
	aa = aa->mean();
	Flux *d_loss = *aa * -1.0;*/
	Flux *d_loss = *(*d_real->log() + *(1.0 - *d_fake)->log())->mean() * -1.0;
	Flux *g_loss = *d_fake->log()->mean() * -1.0;
	//d_loss = d_loss->bypass("real loss\n");
	//g_loss = g_loss->bypass("d_fake loss\n");
	agloss2 = d_loss;
	agloss = g_loss;
	/*
	for(auto a : *tcr->trainvar(g_nsc)) {
		printf("%p: %s\n", a, a->fxName);
	}
	printf("----------------------\n");
	for(auto a : *tcr->trainvar(d_nsc)) {
		printf("%p: %s\n", a, a->fxName);
	}*/
	Flux *d_train = dnet->cx_optimizer(d_loss, lr, TRACER(tcr)->optType, tcr->trainvar(d_nsc));
	Flux *g_train = gnet->cx_optimizer(g_loss, lr, TRACER(tcr)->optType, tcr->trainvar(g_nsc));
	agtrain2 = d_train;
	agtrain = g_train;

	trainCount = 0;

	if(name) tcr->endscope();
	return nsc;
}
Algol::Algol(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, bool contraction, sytet af, floatt lr, const bytet *name)
{
	algol(tcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, contraction, af, lr, name);
}
Algol::Algol(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, bool contraction, sytet af, floatt lr, const bytet *name)
{
	algol(ingate->fxTcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, contraction, af, lr, name);
}
Flux *Algol::train(intt *n_train)
{
	/*floatt e, e2;
	if(trainCount > 1) {
		if(exec1 > 0) {
			agtrc->run({ agtrain, agloss });
			e = agloss->at_d2(0);
			if(e > eloss) exec2 = 0;//�̹��� generator ������ ���������� ������ discriminator�� ����ʰ� generator�� ����
			else {
				eloss -= (eloss - e < 0.001 ? eloss - e : 0.001);//�ִ� ������ 0.001�� generator ���� ���� ��ǥ ����.
				if(exec2 == 0) exec2 = -1;//�̹��� generator������ ���Ơ����Ƿ� ������ discriminator���� ����.
			}
		} else printf("generator skip\n");
		if(exec2 > 0) {
			agtrc->run({ agtrain2, agloss2 });
			e2 = agloss2->at_d2(0);
			if(e2 > eloss2) exec1 = 0;//�̹��� discriminator ������ ���������� ������ generator�� ����ʰ� discriminator�� ����
			else {
				eloss2 -= (eloss2 - e2 < 0.001 ? eloss2 - e2 : 0.001);//�ִ� ������ 0.001�� discriminator ���� ���� ��ǥ ����.
				exec1 = -1;//�̹��� discriminator������ ���Ơ����Ƿ� ������ generator���� ����.
			}
		} else printf("descriminator skip\n");
		if(exec1 < 0) exec1 = 1;
		if(exec2 < 0) exec2 = 1;
		if(exec1 == 0 && exec2 == 0) exec2 = 1;
	} else {*/
		agtrc->run({ agtrain, agloss });//gernerator
		agtrc->run({ agtrain2, agloss2 });//discriminator
		//exec1 = exec2 = 1;
		//eloss = agloss->at_d2(0);
		//eloss2 = agloss2->at_d2(0);
	//}
	//trainCount++;

	return agloss;
}
Flux *Algol::predict(Flux **loss_fx)
{
	return agnet->predict(loss_fx);
}
Flux *Algol::loss2(void)
{
	return agloss2;
}
void Algol::accuracy(Flux *&predicts, Flux *&targets, intt discrete_out)
{
	agnet->accuracy(predicts, targets);
}
Flux *Algol::measureAccuracy(void)
{
	return agnet->measureAccuracy();
}
NameScope *Stratus::stratus(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet af, floatt lr, const bytet *name)
{
	Trace *trc = TRACER(tcr);
	NameScope *nsc = nullx, *t_nsc, *s_nsc;

	ichkBound = ichkBound2 = 0;
	endChkBound = endChkBound2 = 0;
	srtar_loss = srrtar_loss = nullx;

	if(trc->stratusOpt < 0) {//recall, Ÿ�ٽ������� �ڱⱸ���н��� �����͸� �Է��ϰ� �ڱⱸ���н�,
		tartrc = sortrc = trc;//������ �Է½������� ������ �����͸� �Է��ϰ� ����.
		snet_ingate = ingate;//�����ϰ��� �ϴ� ������(�� Ÿ�ٽ������� �����ϱ����� �������� ������� �����ڵ�)�� ���������ڵ�
		snet_targate = targate;
		srtar_net = new(tartrc)Generic;
		t_nsc = srtar_net->generic(tartrc, targate, targate, latent_sz, outdiscret, outdiscret,
			embedim, -1, af, lr, "target");//target auto encoder, Ÿ�ٽ������� �ڱⱸ�� �н��� ����
		return t_nsc;
	}
	if(indiscret > 0 && trc->stratusOpt < 2) trc->stratusOpt = 2;//��½������� �̻굥�����̸� srsor_net->cypred�� 
	//����ϱ����� argmax�� �ϰ� argmax�� ���⸦ ���Ҽ����� ������ ���������Ƿ� 2�� Ÿ������ ���� �����Ҽ�����.
	if(trc->endecode && trc->stratusOpt == 3) trc->stratusOpt = 2;//endecode�� stratusOpt == 3 ���̽��� �� �ʿ����.
	tartrc = trc;//stratusOpt 1,2�� ���̽��� Ÿ�� trc�� �÷����� ���� ���ϵǹǷ� Ÿ�� trc�� �� trc�� �Ѵ�.
	if(trc->trcpart) {
		sortrc = trace(0);
		TRACER(sortrc)->migopt(trc);
	} else sortrc = trc;

	srsor_net = new(sortrc)Generic;
	srtar_net = new(tartrc)Generic;
	if(trc->stratusOpt == 3) {
		srrtar_net = new(tartrc)Generic;
		srrtar_net->makeAnet(tartrc, latent_sz, af);
	}
	if(name) nsc = sortrc->namescope(name);
	if(trc->trcpart) {
		snet_ingate = ingate->partition(sortrc);
		snet_targate = targate->partition(sortrc);
	} else {
		snet_ingate = ingate;
		snet_targate = targate;
	}
	if(trc->stratusOpt < 2) {//source -> target -> linknet -> Ÿ�ٽ������� ���ڵ� ���� �����ڵ�
		srnet3 = new(sortrc)Generic;//link net
		t_nsc = srtar_net->generic(tartrc, targate, targate, latent_sz, outdiscret, outdiscret,
			embedim, -1, af, lr, "target");//target auto encoder
		s_nsc = sortrc->namescope("source_link");
		srsor_net->generic(sortrc, snet_ingate, snet_targate, latent_sz, indiscret, 0,
			embedim, 1, af, lr, "source");
		srnet3->compress(sortrc, srsor_net->cypred, latent_sz, 0, 0, af, lr, "linkage");//��.link net
		sortrc->endscope();

		if(trc->stratusOpt2 == 0) {//partition()�� ���� �׷����� ������ ������� �ʴ´�.
			//srsor_loss = srnet3->zcodec2->sigmoid()->softmaxCrossEntropy(aa)->mean();//target link loss
			//srsor_loss = srnet3->zcodec2->sigmoid()->softmaxCrossEntropy(srtar_net->zcodec2->partition()->sigmoid())->mean();//target link loss
			if(trc->intergate) tarout = new(sortrc)Flux(sortrc, srtar_net->zcodec2->fshape, srtar_net->zcodec2->fdim, srtar_net->zcodec2->qType, variable);
			else tarout = srtar_net->zcodec2->partition(sortrc);
			srsor_loss = srnet3->zcodec->softmaxCrossEntropy(tarout->softmax())->mean();//target link loss
		} else {
			if(trc->intergate) tarout = new(sortrc)Flux(sortrc, srtar_net->zcodec->fshape, srtar_net->zcodec->fdim, srtar_net->zcodec->qType, variable);
			else tarout = srtar_net->zcodec->partition(sortrc);
			srsor_loss = srnet3->zcodec->meanSquareError(tarout);//target link loss, ���� ���� �ڵ� ���� �н�
		}
	} else {//source �� ���ڴ��� Ÿ���� ���ڴ��� �ٷ� ����
		NameScope *rt_nsc;
		t_nsc = srtar_net->generic(tartrc, targate, targate, latent_sz, outdiscret, outdiscret,
			embedim, -1, af, lr, "target");//target auto encoder, Ÿ�ٽ������� �ڱⱸ�� �н��� ����
		s_nsc = srsor_net->compress(sortrc, snet_ingate, latent_sz, indiscret, embedim, af, lr, "source");
		if(trc->stratusOpt == 3) {//Ÿ���� �ڱⱸ���н��� ���ݺ� ���ڵ� �κ�(���� ���� �����ڵ�)�� �Է������Ͽ� �Ĺݺ� ���ڵ� 
			rt_nsc = tartrc->namescope("rtarget");//�κи��� decode2�� �н���Ų��.�� ���ݺε��� �Ĺݺε��� ������ �������� ���·� �н���Ų��.
			srrtar_net->setFrontendFactor(srtar_net, srtar_net->zcodec->partition(tartrc));//srsts.
			srrtar_net->decompress(tartrc, targate->fshape[1], 1);
			srrtar_net->connect(targate, outdiscret, 1, -1, -1);
			tartrc->endscope();
			srrtar_net->coptrain = srrtar_net->cx_optimizer(srrtar_net->closs, TRACER(tartrc)->learningRate > 0 ?
				TRACER(tartrc)->learningRate : -1, TRACER(tartrc)->optType, tartrc->trainvar(rt_nsc));
		}//���� �Է����κ��� Ÿ�ٽ������� �ڱⱸ���н��� ���� ���� �����ڵ带 Ÿ������ ���� �� ����
		if(trc->stratusOpt2 == 0) {
			if(trc->intergate) tarout = new(sortrc)Flux(sortrc, srtar_net->zcodec2->fshape, srtar_net->zcodec2->fdim, srtar_net->zcodec2->qType, variable);
			else tarout = srtar_net->zcodec2->partition(sortrc);
			srsor_loss = srsor_net->zcodec->softmaxCrossEntropy(tarout->softmax())->mean();
		} else {
			if(trc->intergate) tarout = new(sortrc)Flux(sortrc, srtar_net->zcodec->fshape, srtar_net->zcodec->fdim, srtar_net->zcodec->qType, variable);
			else tarout = srtar_net->zcodec->partition(sortrc);
			srsor_loss = srsor_net->zcodec->meanSquareError(tarout);//target link loss, ���� ���� �ڵ� ���� �н�
		}
	}
	srsor_train = srsor_net->cx_optimizer(srsor_loss, lr, TRACER(sortrc)->optType, sortrc->trainvar(s_nsc));//source loss
	if(name) sortrc->endscope();
	return nsc;
}
Stratus::Stratus(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet af, floatt lr, const bytet *name)
{
	stratus(tcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, af, lr, name);
}
Stratus::Stratus(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet af, floatt lr, const bytet *name)
{
	stratus(ingate->fxTcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, af, lr, name);
}
Flux *Stratus::train(intt *n_train)
{
	Trace *tcr = TRACER(tartrc);
	intt n_learn_stop = tcr->nlearnStop < 0 ? tcr->nlearnStop * -1 : tcr->nlearnStop;

	if(tcr->stratusOpt >= 2 && n_learn_stop) {//Ÿ�ٽ����� �ڱⱸ���н��Ŀ� ���̻� �Ʒ������ʰ� Ÿ�ٽ������� 
												//���ڵ� ��Ʈ�� ���������� �����Ͽ� ������ �Ʒ�
		if(srtar_loss && endChkBound == 0) {//Ÿ�ٽ����� �ڱⱸ���н��� ���۵��� �Ϸ���� ���� �����̸� �Ϸ� üũ
			if(srtar_loss->at_d(0) > tcr->elearnStop) ichkBound = 0;
			else ichkBound++;
			if(ichkBound > n_learn_stop) endChkBound = 1;//�����ؼ� nlearnStop�� elearnStop���� ���������̸� 
		}													//���̻� Ÿ���� �ڱ� �����н��� ���� �ʴ´�.
		if(endChkBound) {//Ÿ�ٽ����� �ڱ� �����н��� �Ϸ������ Ÿ�ٽ����� ���ڵ�2 �н� �� �ҽ� ���� �н��� ����.
			if(tcr->stratusOpt == 3 && srrtar_loss && endChkBound2 == 0) {//Ÿ�ٽ����� ���ڵ� ���ڵ�2 �и��н��̸�
				if(srrtar_loss->at_d(0) > tcr->elearnStop) ichkBound2 = 0;//���ڵ�2 �н� �Ϸ� üũ
				else ichkBound2++;
				if(ichkBound2 > n_learn_stop) endChkBound2 = 1;//�����ؼ� nlearnStop�� elearnStop���� ���������̸� 
			}													//���̻� Ÿ�� �Ĺݺ� ���ڵ�2 �н��� ���� �ʴ´�.
			if(tcr->stratusOpt != 3 || endChkBound2) {//Ÿ�� ���ڵ� ���ڵ� �и��н��� �ƴϰų� Ÿ�� ���ڵ�2 �н��� �Ϸ��̸�
				if(tcr->stratusOpt2) tartrc->run({ srtar_net->zcodec });//�ҽ��� Ÿ�� ���� ���� �����ڵ���� ���� �н�.
				else tartrc->run({ srtar_net->zcodec2 });

				if(tcr->intergate) {
					if(tcr->stratusOpt2) tarout->feedf(srtar_net->zcodec);
					else tarout->feedf(srtar_net->zcodec2);
				}
				sortrc->run({ srsor_train, srsor_loss });//source, link train
			} else if(tcr->nlearnStop < 0) {//Ÿ�ٽ����� �ڱ� �����н��� �Ϸ�� ���¿��� ���ڵ�2�� �Ϸ���� �ʾҴµ�
				tartrc->run({ srtar_net->zcodec });//�ҽ� ���� ���� �н� �ɼ��̸� ���ڵ�2�� �ҽ����� �н� ����.
				goto LB2;
			} else {//Ÿ�� ���ڵ� ���ڵ� �и��н��̰� �ҽ� ���� ���� �н��� �ƴϰ� Ÿ�� ���ڵ�2 ��Ʈ �н����̸� 
				srrtar_loss = srrtar_net->train();//Ÿ�� ���ڵ� �н��� ����.
				return srrtar_loss;
			}
		} else if(tcr->nlearnStop < 0) {//Ÿ�ٽ����� �ڱ� �����н� �Ⱓ�߿� �ҽ� ���� �н� �� ���ڵ�2 �н� ����. 
			//Ư��, sign curve�� ���� �ֱ������� �ݺ��Ǵ� �����ʹ� �ֱ� ������ �°� Ÿ�� �н��� ���� ��� ����
			goto LB1;// �ҽ� �н��� ���� �ش��� Ÿ�ٸ� �н��� Ÿ�ٿ� ���� �н��ؾ� �ϴ� �� ����.
		} else {//Ÿ�ٽ����� �ڱ� �����н��� ����
			srtar_loss = srtar_net->train();//target auto encoder train
			return srtar_loss;
		}
	} else {//Ÿ�ٽ����� �ڱ� �����н��� �ҽ� ���� �н��� ������ �Բ� ����
LB1:;	srtar_loss = srtar_net->train();//target auto encoder train, Ÿ�ٽ������� �ڱⱸ�� �н�
		if(tcr->stratusOpt < 0) return srtar_loss;//�ڱⱸ���н��� ����, �������� ���� ��
LB2:;	if(tcr->stratusOpt == 3) srrtar_loss = srrtar_net->train();//Ÿ�ٽ������� ���ڵ�2 �н�
		if(TRACER(sortrc)->intergate) {
			if(TRACER(sortrc)->stratusOpt2) tarout->feedf(srtar_net->zcodec);
			else tarout->feedf(srtar_net->zcodec2);
		}
		sortrc->run({ srsor_train, srsor_loss });//�ҽ��������κ��� Ÿ�� ���ڵ� ��Ʈ �������� �����ڵ带 Ÿ������ �����н�.
	}
	return srsor_loss;
}
Flux *Stratus::predict(Flux **loss_fx)
{
	if(TRACER(sortrc)->stratusOpt == 0) {//source trc�� �÷����� ���ϵǳ� ���ϵǴ� �÷����� Ÿ��trc�� 
		return srsor_net->predict(loss_fx);//���۷��̼��� �� ���� ���̹Ƿ� ������.
	} else {//stratusOpt == 1 | 2 | 3, -1
		Generic *r_net;
		if(TRACER(sortrc)->stratusOpt == 1) {
			r_net = srtar_net;
			sortrc->run({ srnet3->zcodec });
			r_net->zcodec->copyf(srnet3->zcodec);
		} else if(TRACER(sortrc)->stratusOpt > 1) {
			sortrc->run({ srsor_net->zcodec });
			srtar_net->zcodec->copyf(srsor_net->zcodec);//������ srrtar_net->zcodec�� �ҽ����� ����� �����ؾ� �ϴ�
			//���ε� srsts)���� ��Ƽ�� �������� ���Ͽ� �׶��� �����߿� srtar_net->zcodec���� srrtar_net->zcodec�� 
			//����ǹǷ� ���⼭ srrtar_net->zcodec�� �ҽ����� ��°��� ������ ���� srrtar_net->zcodec�� ��������Ʈ �ǹǷ�
			//srrtar_net->zcodec�� �ҽ����� ��� ���� �����Ѵ�.
			if(TRACER(sortrc)->stratusOpt == 2) r_net = srtar_net;
			else r_net = srrtar_net;//3
		} else {//-1, recall, �Է½������� �ڱⱸ���н��� ����� ���� ��.
			srtar_net->zcodec->copyf(snet_ingate);
			r_net = srtar_net;
		}
		r_net->zcodec->reentrance(1);
		auto pred = r_net->predict(loss_fx);//Ÿ�� closs�� ���� �����Ҷ� �����Ǳ� ������ �� endChkBound�� �����Ǿ�
		r_net->zcodec->reentrance(0);		//Ÿ�� �ڱⱸ���н��� ���̻� ������� �ʴ���� �ҽ������н� �����߿� 
		return pred;							//���� ������ �����ϸ� Ÿ�� �ν��� ��� �پ��� ������ ���´�(����)
	}
}
Flux *Stratus::loss2(void)
{
	return srtar_loss;
}
void Stratus::accuracy(Flux *&predicts, Flux *&targets, intt discrete_out)
{
	srtar_net->accuracy(predicts, targets, ((Anet *)srtar_net->canet)->yDiscrete ? 1 : -1);
}
Flux *Stratus::measureAccuracy(void)
{
	return srtar_net->measureAccuracy();
}