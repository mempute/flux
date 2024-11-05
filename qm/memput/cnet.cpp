
#include "anet.h"

//indiscret - 언어의 예로 들면 vocabulary size(전체 학습 문장에서 유니크한 전체 단어 갯수), embedim - 단어 임베딩할 사이즈
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
		zcodec2 = anet->encodeOut;//활성함수 전 플럭스
		if(targate == nullx) goto LB1;
	}
	//zcodec - C_ZZ이면 [batch, outsz(n_stride*n_reduce), n_derive], C_XT이면 //[batch, bindbatch, outsz(cross_out)]
	if(dual_code) {
		if(ingate == nullx) makeAnet(trc, latent_sz, af);
		if(trc->dualChain) dcodec = ((Anet *)canet)->chainEncoder(zcodec, trc->bygate, trc->nbyout, indiscret, embedim, -1, latent_sz);
		else dcodec = ((Anet *)canet)->dualEncoder(zcodec, trc->bygate, trc->nbyout, indiscret, outdiscret, embedim, -1, -1, latent_sz);
	} else if(targate) dcodec = anet->decodeGate(zcodec, -1);//C_XT 케이스(시퀀스 압축 없이 바로 한번에 타겟 연결)이므로 출력 사이즈 없이 끝 -1
	//dcodec - C_ZZ이면 [batch, outsz(n_stride*n_reduce확장), n_derive(hidden)], C_XT이면 [batch, outsz(cross_out), bindbatch(hidden)]
	ctarget = targate;
	if(trc->outback) {//스텝 설정인데 듀얼인코더 실행에만 적용되는 것이면 듀얼인코더 빌드될때만 아웃백 적용
		//또는 듀얼인코더가 아닌 일반 인코딩일때 적용되는 옵션이면 듀얼인코더 빌드되지 않을때 아웃백 적용
		if((trc->dualStep && trc->nbyout >= 0) || (trc->dualStep == 0 && trc->nbyout < 0)) {
			if(name) trc->endscope();
			return nsc;
		}
	}
	cypred = anet->outGate(dcodec, outdiscret, targate->fshape[2], clogit, "conet_out", 0);
	//cypred - discrete이면 [batch, outsz, 1], 아니면 [batch, outsz, out_feat]
	//clogit - discrete일때 [batch, out_seq, vocab_sz], 아니면 cypred와 동일
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
{//차원 축소된 잠채코드 생성 혹은 입력을 타겟으로 학습하는 오토인코더 용.
	if(name) tcr->namescope(name);
	makeAnet(tcr, latent_sz, af);
	Anet *anet = (Anet *)canet;
	zcodec = anet->encodeGate(ingate, outsz, indiscret, embedim);
	zcodec2 = anet->encodeOut;//활성함수 전 플럭스
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
	return zcodec->fshape[1];//tcr 디버그 모드와 상관없이 사이즈는 계산됨
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
	byPolar = 1;//인코더와 디코더 두개 모두 구성된다늠 표시, 이것이 아니고 컴프레스만 호출되어 인코더만 있으면 completeConvolve
				//일때 endecode옵션 설정되있어도 컴프레스에서 완전 인코딩(압축)하게 한다.
	if(name) coaxNsp = trc->namescope(name);
	if(ingate && embedim >= 0) compress(tcr, ingate, latent_sz, indiscret, embedim, af, lr);
	if(trc->nbyout >= 0) {//ingate가 주어졌으면 이는 pcode(이전 문맥코드)이다. 입력은 by게이트에
		bool prev_code = embedim < 0 ? 1 : 0;//다이나젠의 csnr)에서 -1로 주어졋으면 다이나젠에서 코드압축됐다는 
		if(trc->sembedn >= 0) embedim = trc->sembedn;//것으로서 이를 이전 코드로 설정, dncs)의 경우면 stracode설정으로
		if(trc->slatent >= 0) latent_sz = trc->slatent; //다이나젠과 위 컴프레스에서 두번 인코딩후 듀얼 인코더에 입력된다.
		if(canet == nullx) makeAnet(tcr, latent_sz, af);//두거나 아님 생락(이경우 by게이트 전체는 타겟(추론)사이즈)
		if(trc->dualChain) dcodec = ((Anet *)canet)->chainEncoder(prev_code ? ingate : zcodec, trc->bygate,
			trc->nbyout, indiscret, embedim, -1, latent_sz);
		else dcodec = ((Anet *)canet)->dualEncoder(prev_code ? ingate : zcodec, trc->bygate,
			trc->nbyout, indiscret, outdiscret, embedim, -1, -1, latent_sz);//embedim < 0면 ingate는
		//dynagen에서 압축된 이전 문맥 압축코드이고 (압축이 derivefeat로 됐으면 듀얼인코더로 동일하게
		//실행하애 한다. 안그러면 feature차원이 맞지 않게 된다.->듀얼인코더에서 조정하므로 상관없다.)
		//embedim이 0이상으로 호출됐으면 zcodec값이 있으면 이는 제너릭망에서 컨볼브로 압축된 이전 문맥코드
	} else {//chatbot을 듀얼 인코더(auto regression)이 아닌 auto encoding으로 실행할면 <s> 시작토큰
		//을 데이터에서 제거하고 실행하면 32길이를 16길이로 한번만 압축후 디코딩 실행 할경우 9만 스텝정도
		//걸리고 시작토큰이 주어진 경우 3만 스텝정도로 학습이 잘된다. 나중에 검토.
		if(targate) decompress(tcr, targate->fshape[1]);
	}
	ctarget = targate;
	if(trc->outback) {//스텝 설정인데 듀얼인코더 실행에만 적용되는 것이면 듀얼인코더 빌드될때만 아웃백 적용
		//또는 듀얼인코더가 아닌 일반 인코딩일때 적용되는 옵션이면 듀얼인코더 빌드되지 않을때 아웃백 적용
		if((trc->dualStep && trc->nbyout >= 0) || (trc->dualStep == 0 && trc->nbyout < 0)) {
			if(name && step >= 0) trc->endscope();
			return coaxNsp;
		}
	}
	if(targate) connect(targate, outdiscret, step, trc->learningRate > 0 ? trc->learningRate : lr, trc->optType);
	if(name && step >= 0) trc->endscope();//step 이 -1이면 위 함수 내부에서 스코프 클로즈
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

void Generic::accuracy(Flux *&predicts, Flux *&targets, intt discrete_out) //정확도 측정 그라프 빌드
{//학습할때의 입력과 타겟 게이트를 사용하지 않고 따로 입출력을 두는 것은 잠재 코드 정확도를 계산할때는 학습 입출력때의 데이터와
	caccuracy = ((Anet *)canet)->calcAccuracy(predicts, targets, discrete_out);//잠재코드의 형태가 틀리기 때문이다.
}
Flux *Generic::measureAccuracy(void) //정확도 측정 그라프 실행.
{
	((Anet *)canet)->tcrAnt->run({ caccuracy });
	return caccuracy;
}
intt Generic::reduction(intt outsz, intt szlat)
{
	if(TRACER(zcodec->fxTcr)->layer_norm == 1) {
		zcodec = zcodec->layer_normal();
		zcodec = zcodec->actf(ACTF_TANH);
	}//zcodec및 zcodec2는 죄종 리덕션의 잠재코드가 저장한다.
	dcodec = zcodec = ((Anet *)canet)->encodeGate(zcodec, outsz, -1, -1, -1, -1, szlat);//다른 망에서 최종 압축 결과를 사용할며면 zcodec를 연결하면 된다.
	//zcodec = zcodec->bypass("aaa\n");
	//dcodec = zcodec;
	zcodec2 = ((Anet *)canet)->encodeOut;//활성함수 전 플럭스
	return zcodec->fshape[1];//tcr 디버그 모드와 상관없이 사이즈는 계산됨
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
	if(im) zcodec = dcodec;//가장 안쪽 en-decode이면 잠재코드 설정.
	return dcodec->fshape[1];//tcr 디버그 모드와 상관없이 사이즈는 계산됨
}
//v - 커널 사이즈, last - 타겟 시퀀스 길이, conv - 컨볼빙 계수
intt first_convolv(intt v, intt last, intt conv)
{
	for(;v * conv < last; v *= conv);
	if(v == last) return conv;//컨볼빙 계수에 정합되면 컨볼빙 계수를 그대로 리턴

	return (last / v < 2 ? conv : last / v);//마지막 수와 타겟의 비가 2 이상이면 이 계수를 첫번째 컨볼빙 계수로 리턴
}
intt Generic::reduceKernel(Tracer *tcr, intt in_sz)
{//커널사이즈 보다 입력 사이즈가 적으면 커널 사이즈를 더 적게 설정한다.
	Trace *trc = TRACER(tcr);
	intt ksz;

	for(ksz = trc->kernelSize; in_sz < ksz; ksz /= 2) {
		if(ksz / 2 == in_sz) {
			ksz /= 2;
			break;
		} else if(ksz / 2 < in_sz) break;//커널 사이즈를 2의 배수의 outer로 설정한다.
	}
	return ksz;
}
NameScope *Generic::compress(Tracer *tcr, Flux *ingate, intt latent_sz, intt indiscret, intt embedim, sytet af, floatt lr, const bytet *name)
{
	Trace *trc = TRACER(tcr);
	intt sz_lowbound, convolv = (trc->convolving > 1 ? trc->convolving * (trc->kernelSize / trc->bandwidth) : trc->convolving);
	NameScope *nsc = nullx;
	//(low bound, convolve, input seq)->(sz_lowbound,compress out sz,,)
	//가급적 sz_lowbound는 reduction의 입력사이즈가 8보다 적게되지않게 한다. 모자르면 제로패딩됨
	//예) (1,4,32)->(8,8), (0.25,4,32)->(2, 8, 2), (0.25,2,32)->(2,16,8,4,2), (0.25,16,32)->(2,2)
	//(1,4,64)->(8,16,8), (0.5,4,64)->(4,16,4), (1,4,16)->(8,8), (1/8,16,16)->(1,1), (0.5,8,16)->(2,2)
	in_sz = ingate->fshape[1];
	if(name) nsc = tcr->namescope(name);
	
	sz_lowbound = reduceKernel(tcr, in_sz);
	intt save_size = trc->kernelSize;
	trc->kernelSize = sz_lowbound;

	if(trc->bandwidth < sz_lowbound) sz_lowbound = trc->bandwidth;

	sz_lowbound *= trc->lowbound;//압축을 8 이하 사이즈도 허용
	if(sz_lowbound == 0) sz_lowbound = 1;
	else if(trc->compbot > 0) sz_lowbound = trc->compbot;
	ist = 0;
	seq_st[ist++] = in_sz;
	if(convolv > 1) in_sz /= convolv;//convolv == 2) in_sz == 16 | convolv == 4) in_sz == 8
	else in_sz /= sz_lowbound;
	if(in_sz < sz_lowbound) in_sz = sz_lowbound;
	//derivefeat == 2 이면 ingate의 시퀀스가 8 ~ 32사이 이면 32로 확장되어 압축결과 4가 리턴된다.
	in_sz = cbuild(tcr, ingate, in_sz, latent_sz, indiscret, embedim, af, lr);//10 * 32 (vocab size)
	if(convolv <= 1) {
		if(name) tcr->endscope();
		trc->kernelSize = save_size;//복원
		return nsc;
	}
	if(in_sz > sz_lowbound) {// && in_sz >= trc->bandwidth) {
		while(1) {
			((Anet *)canet)->kernszAnt = reduceKernel(tcr, in_sz);//in_sz가 커널 사이즈보다 적제되면
			seq_st[ist++] = in_sz;	//커널사이즈가 축소 조정되게, 이번 빌드과정에서만 사용되므로 복원필요없다.
			in_sz /= convolv;
			if(in_sz < sz_lowbound) in_sz = sz_lowbound;
			//if(trc->ln1) zcodec = zcodec->layer_normal();
			tcr->namescope(ist);
			in_sz = reduction(in_sz);//convolv == 2) in_sz == 16, 8 | convolv == 4) in_sz == 8
			tcr->endscope();
			if(in_sz <= sz_lowbound) break;//==조건에서 <= 조건으로 수정, 나중에 검토
		}
	}
	trc->kernelSize = save_size;//복원
	//lowbound가 1 이상이면 커널 사이즈 끝까지 컨볼빙을 안하는 옵션이므로 completeConvolve(최종출력사이즈가 로우바운드사이즈와
	//같게됐을때 이를 한번더 혼합하는 옵션)가 의미없으므로 실해않한다.
	//byPolar == 0이면 인코딩만 구성된 네트워크, 1이면 인코딩과 디코딩이 모두 구성된 네트워크로서 completeConvolve가 1이면
	//인코딩만 구성된 네트워크만 완전 압축하고 2이면 인코딩과 디코딩이 모두 구성된 네트워크에서 인코딩할때 완정 압축한다.
	//nxcicp.decode gate2이면 디코드 과정에서 압축후 디코딩되므로 여기 인코딩에서 완전압축하지않고 디코딩과정에서 완전압축되게 한다.
	if(trc->endecode == 0 && trc->lowbound <= 1 && ((byPolar == 0 && trc->completeConvolve) || (byPolar && trc->completeConvolve == 2))) {//in_sz == 8,  
		//if(trc->ln2) zcodec = zcodec->layer_normal();
		tcr->namescope("end");
		//spotOpt == 3으로서 최종압축일때만 적용되는 옵션일때 attsign == 1이면 인코딩만 구성된 네트워크에만 최종 압축에서 어텐션 
		//적용하고 인코딩과 디코딩이 모두 구성네트워크의 최종 압축에서 어텐션을 적용 안하게한다. spotOpt가 3이 아니면 attsign == 1는 의미없다.
		in_sz = reduction(in_sz, trc->attsign == 1 && byPolar ? (trc->spotlat > 0 ? trc->spotlat * -1 : 0) : trc->spotlat);//residual reduction수행
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
	//for(intt i = 0;i < ist; i++) seq_st[i] = from->seq_st[i];//현재는 이 변수는 필요없으니 안한다.
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
			intt cv = first_convolv(in_sz, tar_sz, convolv);//convolv에 정합되지 않으면 첫번째는 최소 정합수를 곱한다.
			for(;in_sz != tar_sz;) {
				if(tar_sz > trc->bandwidth && in_sz * cv > tar_sz) {//타겟이 커널 사이즈보다 적으면 decompose2에서 
					break;//타겟사이즈에 맞춰지므로 이 경우는 제외하고 타겟 사이즈에 정합되지 않으면 모자르는 사이즈는 밑으로가서 맞춘다.
				}
				in_sz = (in_sz * cv > tar_sz ? tar_sz : in_sz * cv);
				//if(ln) {
				//	dcodec = dcodec->layer_normal();
				//	if(trc->ln3 == 0) ln = 0;
				//}
				tcr->namescope(ist--);
				in_sz = decompose2(in_sz, 0);
				tcr->endscope();
				cv = convolv;//원 값으로 복원
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
		decompose(tar_sz);//위 싫행에서 in_sz가 tar_sz보다 크게되는 경우는 없다.입력이 in_sz가 더 커서 위 디컴포즈가
		tcr->endscope();	//실행되지 않은 경우는 in_sz가 더 크게되고 어찌됐던 여기 디컴포즈 호출에서 타겟사이즈에 맞춰진다.
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
	closs = anet->calcLoss(clogit, targate);//일방향 추론코드(<s> + 타겟) 타겟 연결, targate - 타겟 + <e>
	bloss = anet->batchLoss;
	//caccuracy = anet->calcAccuracy(cypred, targate);
	if(step == 1) return;

	if(step < 0) {
		anet->tcrAnt->endscope();//현 망의 가중치 변수만 학습시키기위해 네임스코프를 닫는다.
		coptrain = cx_optimizer(closs, lr, opt_t, anet->tcrAnt->trainvar(coaxNsp));
	} else coptrain = cx_optimizer(closs, lr, opt_t);
}
Flux *Generic::train(intt *n_train)
{
	Anet *anet = (Anet *)canet;
	Trace *trc = TRACER(anet->tcrAnt);

	if(trc->eternity) {//영속코드 학습 실행
		anet->teternal->reentrance(0);//영속코드 반영쪽 코드실행하게 한다.
		anet->reternal->reentrance(1);//영속코드를 회상쪽 코드 실행 안하게한다.
		trc->run({ coptrain, closs, anet->ovCode });//ovCode는 ppcode복사
	} else trc->run({ coptrain, closs });
	
	return closs;
}
Flux *Generic::_predict(Flux *robjec, Flux **loss_fx) //타겟이 학습할때와 틀리면 가중치 로드를 하므로 학습후 이어서 바로 예측할 경우
{									//이번에 학습한 것이 있으면 가중치 저장을 꼭 먼저 해야한다.->loadedWeight처리로 괜찬다.
	Anet *anet = (Anet *)canet;
	Trace *tcr = TRACER(anet->tcrAnt);

	if(tcr->eternity == 1) {//영속코드 실행이고 추론때는 영속코드 반영은 안하는 경우
		anet->teternal->reentrance(1);//영속코드 반영쪽 코드실행 안하게한다.
		anet->reternal->reentrance(0);//영속코드를 회상쪽 코드만 실행하게 한다.
	} else if(tcr->eternity == 2) {//영속코드 실행이고 추론때 영속코드 반영하는 경우
		anet->teternal->reentrance(0);//영속코드 반영쪽 코드실행하게 한다.
		anet->reternal->reentrance(1);//영속코드를 회상쪽 코드 실행 안하게한다.
		if(loss_fx) {
			if(bloss) {
				tcr->run({ robjec, bloss, anet->ovCode });//ovCode는 ppcode복사
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
//배치로 수행할거면 배치사이즈를 batchPart보다 적게 하여 배치분할 분산 실행않도록 한다. 
Flux *Generic::inference(Flux **loss_fx)
{
	Trace *tcr = TRACER(((Anet *)canet)->tcrAnt);
	Flux *bgate = tcr->bygate, *go = tcr->gotok, *end = tcr->endtok;
	intt nxin = bgate->fshape[1] - tcr->nbyout;//입력 길이, 없으면 0이고, 처음부터 추론하고 입력은 pcode이다.
	intt width = go->fshape[1];//추론 시작 토큰 길이, 이 길이 단위로 추론
	intt rsz = 0;
	Flux *pred;
	void *p;
	//bgate - 제너릭 정의할때 입력을 주지않고 바이게이트 입력과 <s>+타겟을 쌍으로 입력부터 듀얼인코더로 
	//		학습한 경우 입력과 <s>+타겟 쌍이고 제너릭의 입력이 주어진 경우 <s>+타겟만이며 이때 입력은 
	//		압축되어 듀얼인코더의 pcode로 주입된다. 학습때 입력이 주어진 경우 입력 파트를 적재한 상태에서 
	//		호출해야 한다.	 bygate에 직접 적재하므로 따로 feed함수 호출안는다. 따라서 feed로 적재하지 
	//		않으므로 배치로 수행해도 배치분할 분산 실행않도록 큰수를 설정한다.
	if(nxin < 0) throwFault(-1, "not auto regression\n");
	if(bgate->fshape[0] != go->fshape[0]) go->resizing2(bgate, "inference");
	if(bgate->fshape[0] != end->fshape[0]) end->resizing2(bgate, "inference");
	for(intt i = 0; i < bgate->fshape[0]; i++) {//시작 토큰 초기 주입, netshell, dual chatbot
		for(intt k = 0; k < width; k++) {		//소스에서와 같이 외부에서 주입하므로 필요없다.
			p = go->read_p({ i, k }, &rsz);		//나중에 제거
			bgate->write_p({ i, nxin + k }, p, rsz);
		}
	}
/*//실제 추론은 실행하지 않고 seq to seq 작동 테스트
bgate->printo(2, 3);
Flux *dd = flux(tcr, { bgate->fshape[0], 15, 1 }, tfloat, variable);//아래 32보다 작게하면 
dd->arange(-1);								//end token까지만 실행되어 배치 1개씩 실행 테스트
pred = flux(tcr, { bgate->fshape[0], 32, 1 }, tfloat, variable);
pred->fill(0.0);//reset
pred->howrite(dd);//target
end = flux(tcr, { bgate->fshape[0], 1, 1 }, tfloat, variable);//token
end->fill(888.0);
pred->howrite(end, 15);//input*/
	intt j = nxin;//inplus이면 입력 + 예측 값이 출력되므로 예측값부터 추출하기위해 시작값을 nxin으로 설정후 0로 리셋한다.
	if(tcr->inplus) nxin = 0;//inplus는 학습때 설정하여 모델을 빌드하고 추론때는 inplus구성없이 모델빌드후 추론해도 된다.
	for(;;) {//입력과 타겟을 쌍으로 입력부터 듀얼인코더로 학습한 경우 입력길이를 제외한 타겟부터 추론
		pred = _predict(cypred, loss_fx);//그냥 단순히 마자막 예측의 로스가 리턴된다.
		if(bgate->fshape[0] == 1 && !memcmp(pred->read_p({ 0, j - nxin }), end->begin_p(), end->sizefx())) {
			break;//배치가 1개일 경우 종료 토큰이 추론되면 종료
		}
		j += width;
		if(j >= bgate->fshape[1]) break;//모든 배치의 시퀀스 끝까지 추론됐으면 종료
		for(intt i = 0; i < pred->fshape[0]; i++) {//이번 시퀀스 번째의 추론을 모든 배지에서 발췌하여 다음 시퀀스 입력으로 주입
			for(intt k = 0; k < width; k++) {
				p = pred->read_p({ i, j - width - nxin + k }, &rsz);
				bgate->write_p({ i, j + k }, p, rsz);//바이게이트로 시퀀스를 진행하며 쿼리
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
	
	if(contraction) {//입력과 출력이 길이 대칭, 입력을 압축후 디코딩 출력
		g_nsc = gnet->generic(tcr, ingate, targate, latent_sz, indiscret, outdiscret, embedim, 1, af, lr, "generator");
	} else {//입력이 압축코드, 바로 디코딩 출력
		g_nsc = tcr->namescope("generator");
		gnet->makeAnet(tcr, latent_sz, af);
		gnet->setFrontendFactor2(ingate);
		gnet->decompose(targate->fshape[1]);
		gnet->connect(targate, outdiscret, 1, tcr->learningRate > 0 ? tcr->learningRate : lr, tcr->optType);
		tcr->endscope();
	}
	Flux *gen_out;
	if(outdiscret) {//generator가 이산일 경우 미분되지 않는 argmax대신
		gen_out = gnet->clogit->softmax();//미분가능한 소프트맥스로 대체
	} else gen_out = gnet->cypred;
	NameScope *d_nsc = tcr->namescope("discriminator", 1);//아래 두번째 컴프레스에서 첫번째 컴프레스의
	//가중치를 공유하기위해 reuse 1로 설정. 의미적으로 생성자의 예측값은 타겟값이고 out discret는 타겟의 
	//가지수이고 타겟 입력을 압축하는 것이므로 out discret를 입력가지 수로 설정한다.
	//gen_out - [batch, target_seq, target_feat]
	dnet->compress(tcr, gen_out, latent_sz, 0, 0, af, lr);
	Flux *d_fake = dnet->zcodec->reshape({ -1, dnet->zcodec->fshape[1] * dnet->zcodec->fshape[2] })->layer_dense(1, "sigmoid", Initializer::xavier, "d_dense");
	if(outdiscret) {
		targate = targate->squeeze(2);//끝에 1인 차원을 없앤다.
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
			if(e > eloss) exec2 = 0;//이번에 generator 오차가 증가됐으면 다음에 discriminator은 실행않고 generator민 수행
			else {
				eloss -= (eloss - e < 0.001 ? eloss - e : 0.001);//최대 감소폭 0.001로 generator 오차 감소 목표 설정.
				if(exec2 == 0) exec2 = -1;//이번에 generator오차가 낮아졋으므로 다음에 discriminator실행 설정.
			}
		} else printf("generator skip\n");
		if(exec2 > 0) {
			agtrc->run({ agtrain2, agloss2 });
			e2 = agloss2->at_d2(0);
			if(e2 > eloss2) exec1 = 0;//이번에 discriminator 오차가 증가됐으면 다음에 generator은 실행않고 discriminator민 수행
			else {
				eloss2 -= (eloss2 - e2 < 0.001 ? eloss2 - e2 : 0.001);//최대 감소폭 0.001로 discriminator 오차 감소 목표 설정.
				exec1 = -1;//이번에 discriminator오차가 낮아졋으므로 다음에 generator실행 설정.
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

	if(trc->stratusOpt < 0) {//recall, 타겟시퀀스에 자기구조학습할 데이터를 입력하고 자기구조학습,
		tartrc = sortrc = trc;//예측때 입력시퀀스에 복원할 데이터를 입력하고 에측.
		snet_ingate = ingate;//복원하고자 하는 시퀀스(현 타겟시퀀스를 예측하기위애 상위에서 만들어진 압축코드)의 최종압축코드
		snet_targate = targate;
		srtar_net = new(tartrc)Generic;
		t_nsc = srtar_net->generic(tartrc, targate, targate, latent_sz, outdiscret, outdiscret,
			embedim, -1, af, lr, "target");//target auto encoder, 타겟시퀀스의 자기구조 학습망 생성
		return t_nsc;
	}
	if(indiscret > 0 && trc->stratusOpt < 2) trc->stratusOpt = 2;//출력시퀀스가 이산데이터이면 srsor_net->cypred을 
	//출력하기위해 argmax를 하고 argmax는 기울기를 구할수없어 역전파 되지않으므로 2번 타입으로 맊에 수행할수없다.
	if(trc->endecode && trc->stratusOpt == 3) trc->stratusOpt = 2;//endecode면 stratusOpt == 3 케이스로 할 필요없다.
	tartrc = trc;//stratusOpt 1,2번 케이스가 타겟 trc의 플럭스로 예측 리턴되므로 타겟 trc를 주 trc로 한다.
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
	if(trc->stratusOpt < 2) {//source -> target -> linknet -> 타겟시퀀스의 인코딩 최종 잠재코드
		srnet3 = new(sortrc)Generic;//link net
		t_nsc = srtar_net->generic(tartrc, targate, targate, latent_sz, outdiscret, outdiscret,
			embedim, -1, af, lr, "target");//target auto encoder
		s_nsc = sortrc->namescope("source_link");
		srsor_net->generic(sortrc, snet_ingate, snet_targate, latent_sz, indiscret, 0,
			embedim, 1, af, lr, "source");
		srnet3->compress(sortrc, srsor_net->cypred, latent_sz, 0, 0, af, lr, "linkage");//ㄱ.link net
		sortrc->endscope();

		if(trc->stratusOpt2 == 0) {//partition()의 후행 그래프는 역전파 실행되지 않는다.
			//srsor_loss = srnet3->zcodec2->sigmoid()->softmaxCrossEntropy(aa)->mean();//target link loss
			//srsor_loss = srnet3->zcodec2->sigmoid()->softmaxCrossEntropy(srtar_net->zcodec2->partition()->sigmoid())->mean();//target link loss
			if(trc->intergate) tarout = new(sortrc)Flux(sortrc, srtar_net->zcodec2->fshape, srtar_net->zcodec2->fdim, srtar_net->zcodec2->qType, variable);
			else tarout = srtar_net->zcodec2->partition(sortrc);
			srsor_loss = srnet3->zcodec->softmaxCrossEntropy(tarout->softmax())->mean();//target link loss
		} else {
			if(trc->intergate) tarout = new(sortrc)Flux(sortrc, srtar_net->zcodec->fshape, srtar_net->zcodec->fdim, srtar_net->zcodec->qType, variable);
			else tarout = srtar_net->zcodec->partition(sortrc);
			srsor_loss = srnet3->zcodec->meanSquareError(tarout);//target link loss, 최종 잠재 코드 연결 학습
		}
	} else {//source 의 인코더와 타겟의 디코더를 바로 연결
		NameScope *rt_nsc;
		t_nsc = srtar_net->generic(tartrc, targate, targate, latent_sz, outdiscret, outdiscret,
			embedim, -1, af, lr, "target");//target auto encoder, 타겟시퀀스의 자기구조 학습망 생성
		s_nsc = srsor_net->compress(sortrc, snet_ingate, latent_sz, indiscret, embedim, af, lr, "source");
		if(trc->stratusOpt == 3) {//타겟의 자기구조학습의 전반부 인코딩 부분(최종 압축 잠재코드)를 입력으로하여 후반부 디코딩 
			rt_nsc = tartrc->namescope("rtarget");//부분만을 decode2로 학습시킨다.즉 전반부따로 후반부따로 양측을 완정압축 형태로 학습시킨다.
			srrtar_net->setFrontendFactor(srtar_net, srtar_net->zcodec->partition(tartrc));//srsts.
			srrtar_net->decompress(tartrc, targate->fshape[1], 1);
			srrtar_net->connect(targate, outdiscret, 1, -1, -1);
			tartrc->endscope();
			srrtar_net->coptrain = srrtar_net->cx_optimizer(srrtar_net->closs, TRACER(tartrc)->learningRate > 0 ?
				TRACER(tartrc)->learningRate : -1, TRACER(tartrc)->optType, tartrc->trainvar(rt_nsc));
		}//이하 입력으로부터 타겟시퀀스의 자기구조학습의 최종 압축 잠재코드를 타겟으로 연결 망 생성
		if(trc->stratusOpt2 == 0) {
			if(trc->intergate) tarout = new(sortrc)Flux(sortrc, srtar_net->zcodec2->fshape, srtar_net->zcodec2->fdim, srtar_net->zcodec2->qType, variable);
			else tarout = srtar_net->zcodec2->partition(sortrc);
			srsor_loss = srsor_net->zcodec->softmaxCrossEntropy(tarout->softmax())->mean();
		} else {
			if(trc->intergate) tarout = new(sortrc)Flux(sortrc, srtar_net->zcodec->fshape, srtar_net->zcodec->fdim, srtar_net->zcodec->qType, variable);
			else tarout = srtar_net->zcodec->partition(sortrc);
			srsor_loss = srsor_net->zcodec->meanSquareError(tarout);//target link loss, 최종 잠재 코드 연결 학습
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

	if(tcr->stratusOpt >= 2 && n_learn_stop) {//타겟시퀀스 자기구조학습후에 더이상 훈련하지않고 타겟시퀀스의 
												//인코딩 파트의 최종압촉을 고정하여 나머지 훈련
		if(srtar_loss && endChkBound == 0) {//타겟시퀀스 자기구조학습이 시작된후 완료되지 않은 상태이면 완료 체크
			if(srtar_loss->at_d(0) > tcr->elearnStop) ichkBound = 0;
			else ichkBound++;
			if(ichkBound > n_learn_stop) endChkBound = 1;//연속해서 nlearnStop번 elearnStop이하 오차에러이면 
		}													//더이상 타겟의 자기 구조학습은 하지 않는다.
		if(endChkBound) {//타겟시퀀스 자기 구조학습이 완료됐으면 타겟시퀀스 디코딩2 학습 및 소스 연결 학습만 수행.
			if(tcr->stratusOpt == 3 && srrtar_loss && endChkBound2 == 0) {//타겟시퀀스 인코딩 디코딩2 분리학습이면
				if(srrtar_loss->at_d(0) > tcr->elearnStop) ichkBound2 = 0;//디코딩2 학습 완료 체크
				else ichkBound2++;
				if(ichkBound2 > n_learn_stop) endChkBound2 = 1;//연속해서 nlearnStop번 elearnStop이하 오차에러이면 
			}													//더이상 타겟 후반부 디코딩2 학습은 하지 않는다.
			if(tcr->stratusOpt != 3 || endChkBound2) {//타겟 인코딩 디코딩 분리학습이 아니거나 타겟 디코딩2 학습도 완료이면
				if(tcr->stratusOpt2) tartrc->run({ srtar_net->zcodec });//소스와 타겟 최종 압축 잠재코드와의 연결 학습.
				else tartrc->run({ srtar_net->zcodec2 });

				if(tcr->intergate) {
					if(tcr->stratusOpt2) tarout->feedf(srtar_net->zcodec);
					else tarout->feedf(srtar_net->zcodec2);
				}
				sortrc->run({ srsor_train, srsor_loss });//source, link train
			} else if(tcr->nlearnStop < 0) {//타겟시퀀스 자기 구조학습은 완료된 상태에서 디코딩2가 완료되지 않았는데
				tartrc->run({ srtar_net->zcodec });//소스 연결 병행 학습 옵션이면 디코딩2와 소스연결 학습 수행.
				goto LB2;
			} else {//타겟 인코딩 디코딩 분리학습이고 소스 연결 병행 학습이 아니고 타겟 디코딩2 파트 학습중이면 
				srrtar_loss = srrtar_net->train();//타겟 디코딩 학습만 수행.
				return srrtar_loss;
			}
		} else if(tcr->nlearnStop < 0) {//타겟시퀀스 자기 구조학습 기간중에 소스 연결 학습 및 디코딩2 학습 수행. 
			//특히, sign curve와 같이 주기적으로 반복되는 데이터는 주기 정도는 맞게 타겟 학습과 같이 어느 정도
			goto LB1;// 소스 학습을 끌어 준다음 타겟만 학습후 타겟에 맟춰 학습해야 하는 것 같음.
		} else {//타겟시퀀스 자기 구조학습만 수행
			srtar_loss = srtar_net->train();//target auto encoder train
			return srtar_loss;
		}
	} else {//타겟시퀀스 자기 구조학습과 소스 연결 학습을 끝까지 함께 수행
LB1:;	srtar_loss = srtar_net->train();//target auto encoder train, 타겟시퀀스의 자기구조 학습
		if(tcr->stratusOpt < 0) return srtar_loss;//자기구조학습만 수행, 예측때는 복원 평가
LB2:;	if(tcr->stratusOpt == 3) srrtar_loss = srrtar_net->train();//타겟시퀀스의 디코딩2 학습
		if(TRACER(sortrc)->intergate) {
			if(TRACER(sortrc)->stratusOpt2) tarout->feedf(srtar_net->zcodec);
			else tarout->feedf(srtar_net->zcodec2);
		}
		sortrc->run({ srsor_train, srsor_loss });//소스시퀀스로부터 타겟 인코딩 파트 최종압축 잠재코드를 타겟으로 연결학습.
	}
	return srsor_loss;
}
Flux *Stratus::predict(Flux **loss_fx)
{
	if(TRACER(sortrc)->stratusOpt == 0) {//source trc의 플럭스가 리턴되나 리턴되는 플럭스를 타겟trc로 
		return srsor_net->predict(loss_fx);//오퍼레이션할 일 없을 것이므로 괜찬다.
	} else {//stratusOpt == 1 | 2 | 3, -1
		Generic *r_net;
		if(TRACER(sortrc)->stratusOpt == 1) {
			r_net = srtar_net;
			sortrc->run({ srnet3->zcodec });
			r_net->zcodec->copyf(srnet3->zcodec);
		} else if(TRACER(sortrc)->stratusOpt > 1) {
			sortrc->run({ srsor_net->zcodec });
			srtar_net->zcodec->copyf(srsor_net->zcodec);//원래는 srrtar_net->zcodec에 소스넷의 출력을 복사해야 하는
			//것인데 srsts)에서 파티션 설정으로 인하여 그라프 실행중에 srtar_net->zcodec값이 srrtar_net->zcodec에 
			//복사되므로 여기서 srrtar_net->zcodec에 소스넷의 출력값을 복사해 봐야 srrtar_net->zcodec이 오버라이트 되므로
			//srrtar_net->zcodec에 소스넷의 출력 값을 복사한다.
			if(TRACER(sortrc)->stratusOpt == 2) r_net = srtar_net;
			else r_net = srrtar_net;//3
		} else {//-1, recall, 입력시퀀스를 자기구조학습한 결과로 복원 평가.
			srtar_net->zcodec->copyf(snet_ingate);
			r_net = srtar_net;
		}
		r_net->zcodec->reentrance(1);
		auto pred = r_net->predict(loss_fx);//타겟 closs가 여기 예측할때 설정되기 때문에 위 endChkBound가 설정되어
		r_net->zcodec->reentrance(0);		//타겟 자기구조학습은 더이상 실행되지 않더라고 소스연결학습 수행중에 
		return pred;							//여기 에측을 수행하면 타겟 로스가 계속 줄어드는 것으로 나온다(정상)
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