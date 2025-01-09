#include <vector>
#include <tuple>
#include <sstream>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <unistd.h>
#include <sys/stat.h>
#include <float.h>
#include <math.h>
//#include <Python.h>


#define MAX_CHAR 10
#define MAX_CHAR_PER_LINE 200000


//ref: https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
bool doubleIsEqual(double x, double y,
	double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON)
{
	//Comparison using an absolute value
	double diff = fabs(x - y);
	if (diff <= maxDiff)
		return true;

	//Comparison using a relative value
	x = fabs(x);
	y = fabs(y);
	double largest = (y > x) ? y : x;

	if (diff <= largest * maxRelDiff)
		return true;
	return false;
}

bool doubleIsZero(double x, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	return doubleIsEqual(x, 0.0, maxDiff, maxRelDiff);
}

//x <= y
bool doubleLeq(double x, double y, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	return (doubleIsEqual(x, y, maxDiff, maxRelDiff)) || (x < y);

}

//x >= y
bool doubleGeq(double x, double y, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	return (doubleIsEqual(x, y, maxDiff, maxRelDiff)) || (x > y);
}

int sign(int x) {
	int ret;
	if (x > 0)
		ret = 1;
	else if (x == 0)
		ret = 0;
	else
		ret = -1;
	return ret;
}

int sign(double x, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	if (doubleIsZero(x, maxDiff, maxRelDiff))
		return 0;
	if (x > 0)
		return 1;
	if (x < 0)
		return -1;
}

template <class T>
T min(T x, T y){
	if (x < y)
		return x;
	return y;
}

template <class T>
T max(T x, T y){
	if (x > y)
		return x;
	return y;
}

template <class T>
T min(T* x, int numElm, int stride = 1){
	T val, ret = x[0];
	for (int i = 1; i < numElm; i++){
		val = x[i * stride];
		if (val < ret)
			ret = val;
	}
	return ret;
}

template <class T>
T max(T* x, int numElm, int stride = 1){
	T val, ret = x[0];
	for (int i = 1; i < numElm; i++){
		val = x[i * stride];
		if (val > ret)
			ret = val;
	}
	return ret;
}

template <class T>
void min(T &minVal, int &minIdx, T* x, int numElm, int stride = 1) {
	T val;
	minVal = x[0];
	minIdx = 0;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];
		if (val < minVal) {
			minVal = val;
			minIdx = i;
		}
	}
}

template <class T>
void max(T &maxVal, int &maxIdx, T* x, int numElm, int stride = 1) {
	T val;
	maxVal = x[0];
	maxIdx = 0;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];
		if (val > maxVal) {
			maxVal = val;
			maxIdx = i;
		}
	}
}

void minWithTies(int &minVal, int *minIdxes, int &numMin, int* x, int numElm, int stride = 1) {
	int val;
	minVal = x[0];
	minIdxes[0] = 0;
	numMin = 1;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];
		if (val < minVal) {
			minVal = val;
			minIdxes[0] = i;
			numMin = 1;
		}
		else if (val == minVal) {
			minIdxes[numMin] = i;
			numMin++;
		}
	}
}

void minWithTies(double& minVal, int* minIdxes, int& numMin, double* x, int numElm,
	int stride = 1, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	double val;
	minVal = x[0];
	minIdxes[0] = 0;
	numMin = 1;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];

		if (doubleIsEqual(val, minVal, maxDiff, maxRelDiff)) {
			minIdxes[numMin] = i;
			numMin++;
		}
		else if (val < minVal) {
			minVal = val;
			minIdxes[0] = i;
			numMin = 1;
		}
	}
}

void maxWithTies(int &maxVal, int *maxIdxes, int &numMax, int* x, int numElm, int stride = 1) {
	int val;
	maxVal = x[0];
	maxIdxes[0] = 0;
	numMax = 1;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];
		if (val > maxVal) {
			maxVal = val;
			maxIdxes[0] = i;
			numMax = 1;
		}
		else if (val == maxVal) {
			maxIdxes[numMax] = i;
			numMax++;
		}
	}
}

void maxWithTies(double& maxVal, int* maxIdxes, int& numMax, double* x, int numElm,
	int stride = 1, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	double val;
	maxVal = x[0];
	maxIdxes[0] = 0;
	numMax = 1;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];

		if (doubleIsEqual(val, maxVal, maxDiff, maxRelDiff)) {
			maxIdxes[numMax] = i;
			numMax++;
		}
		else if (val > maxVal) {
			maxVal = val;
			maxIdxes[0] = i;
			numMax = 1;
		}
	}
}

template <class T>
T sum(T *x, int numElm, int stride = 1){
	T total = 0;
	for (int i = 0; i < numElm; i++) {
		total += x[i * stride];
	}
	return total;
}

template <class T>
T sum2(T *x, int numElm, int stride = 1){
	T val, total2 = 0;
	for (int i = 0; i < numElm; i++){
		val = x[i * stride];
		total2 += val * val;
	}
	return total2;
}

template <class T>
T mean(T* x, int numElm, int stride = 1){
	return sum(x, numElm, stride) / numElm;
}

double var(double* x, int numElm, int stride = 1,
	double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON){
	double total2 = sum2(x, numElm, stride);
	double total = sum(x, numElm, stride);
	double varVal = total2 / numElm - (total * total) / (numElm * numElm);
	if (doubleIsZero(varVal, maxDiff, maxRelDiff) || varVal < 0)
		varVal = 0;
	return varVal;
}

double stdv(double* x, int numElm, int stride = 1,
	double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON){
	return sqrt(var(x, numElm, stride, maxDiff, maxRelDiff));
}

void zscore(double *zx, double *x, int numElm, int zstride = 1, int stride = 1,
	double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON){
	double avg = mean(x, numElm, stride);
	double stdev = stdv(x, numElm, stride, maxDiff, maxRelDiff);
	if (doubleIsZero(stdev, maxDiff, maxRelDiff))
		stdev = 1;
	for (int i = 0; i < numElm; i++)
		zx[i * zstride] = (x[i * stride] - avg) / stdev;
}

bool ismember(int x, std::vector<int> v) {
	return std::find(v.begin(), v.end(), x) != v.end();
}

bool isequal(int* x, int* y, int numElm) {
	for (int i = 0; i < numElm; i++) {
		if(x[i] != y[i])
			return false;
	}
	return true;
}

void getNNs(double* nnDists, int* nnInds, double* pDistMtx, int numTrain) {
	double* pDistVec;
	for (int i = 0; i < numTrain; i++) {
		pDistVec = pDistMtx + i * numTrain;
		min(nnDists[i], nnInds[i], pDistVec, numTrain);
	}
}

void update_WK(int* rankedInds_WK, double* minNNDists_WK, int* nnPInds_WK, double* pDistMtx, bool* ranked_WK, bool* validU_WK, int numTrain, int numRanked, int numPLabeled) {
	double* pDistVec, nnDistU, dist, minNNDist = INT_MAX;
	int curInd, nnIndU, minNNInd;
	int nnPInd;
	for (int i = 0; i < numRanked; i++) {
		curInd = rankedInds_WK[i];
		pDistVec = pDistMtx + curInd * numTrain;

		nnDistU = INT_MAX;
		for (int j = 0; j < numTrain; j++) {
			if (!validU_WK[j])
				continue;

			dist = pDistVec[j];
			if (dist < nnDistU) {
				nnIndU = j;
				nnDistU = dist;
			}
		}

		if (nnDistU < minNNDist) {
			minNNInd = nnIndU;
			minNNDist = nnDistU;
			nnPInd = curInd;
		}
	}
	rankedInds_WK[numRanked] = minNNInd;
	minNNDists_WK[numRanked - numPLabeled] = minNNDist;
	nnPInds_WK[numRanked - numPLabeled] = nnPInd;
	ranked_WK[minNNInd] = true;
}



//Li Wei, Eamonn J.Keogh: Semi-supervised time series classification. KDD 2006: 748-753
//This is NOT an accurate re-implementation of WK, and is NOT used in the experiments.
extern "C" int rankTrainInds_WK(int* rankedInds_WK, double* minNNDists_WK, int* nnPInds_WK, double* nnDists, int* nnInds, int* seed,
	double* pDistMtx, bool* ranked_WK, bool* validU_WK, int numTrain, int numPLabeled, int minNumP, int maxNumP) {

	memcpy(rankedInds_WK, seed, numPLabeled * sizeof(int));
	memset(ranked_WK, 0, numTrain * sizeof(bool));
	for (int i = 0; i < numTrain; i++) {
		for (int j = 0; j < numPLabeled; j++) {
			if (i == seed[j]) {
				ranked_WK[i] = true;
				break;
			}
		}
	}

	getNNs(nnDists, nnInds, pDistMtx, numTrain);
	int numValidU;
	for (int i = numPLabeled; i < maxNumP; i++) {
		memset(validU_WK, 0, numTrain * sizeof(bool));
		numValidU = 0;
		for (int j = 0; j < numTrain; j++) {
			if (ranked_WK[j] || !ranked_WK[nnInds[j]])
				continue;
			validU_WK[j] = true;
			numValidU++;
		}

		if (!numValidU && i >= minNumP) {
			return i;
		}
		else {
			if (!numValidU) {
				for (int j = 0; j < numTrain; j++) {
					validU_WK[j] = !ranked_WK[j];
				}
			}
			update_WK(rankedInds_WK, minNNDists_WK, nnPInds_WK, pDistMtx, ranked_WK, validU_WK, numTrain, i, numPLabeled);
		}
	}
	return maxNumP;

}

//Chotirat Ann Ratanamahatana, Dechawut Wanichsan: Stopping Criterion Selection for Efficient Semi-supervised Time Series Classification.Software Engineering, Artificial Intelligence, Networking and Parallel / Distributed Computing 2008: 1-14
extern "C" int sc_RW(double* minNNDists, int minNumP, int maxNumP, int numTrain, int numPLabeled) {

//	for (int i = 0; i < numTrain - numPLabeled; i++){
//		printf("%f ", minNNDists[i]);
//	}
//	printf("\n");
//
//	printf("%d %d %d %d\n", minNumP, maxNumP, numTrain, numPLabeled);

	//initialization
	double minNNDist, sum, sum2;
	sum = sum2 = 0;
	for (int i = 0; i < minNumP - numPLabeled + 1; i++) {
		minNNDist = minNNDists[i];
		sum += minNNDist;
		sum2 += minNNDist * minNNDist;
	}

	double diff, std, scc, maxScc = INT_MIN;
	int preNumP, initNumU = numTrain - numPLabeled;
	for (int i = minNumP - numPLabeled + 1; i < min(maxNumP + 2, numTrain) - numPLabeled; i++) {
		minNNDist = minNNDists[i];
		sum += minNNDist;
		sum2 += minNNDist * minNNDist;

		diff = fabs(minNNDists[i] - minNNDists[i - 1]);
		std = sum2 / (i + 1) - sum * sum / ((i + 1) * (i + 1));
		std = !doubleLeq(std, 0.0) ? sqrt(std) : 1;
		scc = diff / std * (double)(initNumU - i) / initNumU;

		if (scc > maxScc) {
			maxScc = scc;
			preNumP = numPLabeled + i - 1;
		}
	}

//	printf("%d\n", preNumP);

	return preNumP;
}

extern "C" void discretize(int* seq, double* ts, int tsLen, int card) {

	double minVal = min(ts, tsLen);
	double maxVal = max(ts, tsLen);

//	printf("min = %f, max = %f\n", minVal, maxVal);

	if (doubleIsEqual(minVal, maxVal)) {
		for (int i = 0; i < tsLen; i++) {
			seq[i] = 1;
		}
	}
	else {
		for (int i = 0; i < tsLen; i++) {
			seq[i] = round((ts[i] - minVal) / (maxVal - minVal) * (card - 1)) + 1;	//this can lead to incorrect values due to loss of precision
		}
	}

}

extern "C" long long getRdl(int* hypoSeq, long long& cumNumMiss, double* nextTs, int* nextSeq, int numTrain, int numRanked, int tsLen, int card) {
	discretize(nextSeq, nextTs, tsLen, card);
	for (int i = 0; i < tsLen; i++) {
//		printf("%f, %d, %d\n", nextTs[i], nextSeq[i], hypoSeq[i]);
		if (nextSeq[i] != hypoSeq[i])
			cumNumMiss++;
	}

	//needs log2(card) to be an integer
	return (double)(numTrain - numRanked + 1) * tsLen * log2(card)
		+ cumNumMiss * (log2(card) + ceil(log2(tsLen)));	//May require fixing! Can this lead to incorrect results due to loss of precision?
}

//Nurjahan Begum, Bing Hu, Thanawin Rakthanmanon, Eamonn J.Keogh: A Minimum Description Length Technique for Semi-Supervised Time Series Classification. IRI 2013: 171 - 192
extern "C" int sc_BHRK(double* tss, int* rankedInds, int* hypoSeq, int* nextSeq,
	int minNumP, int maxNumP, int numTrain, int numPLabeled, int tsLen, int card) {

//	for (int i = 0; i < numTrain; i++){
//		printf("%d ", rankedInds[i]);
//	}
//	printf("\n%d %d %d %d %d %d\n", minNumP, maxNumP, numTrain, numPLabeled, tsLen, card);


	double* ts;
	int preNumP, optPreNumP;
	long long cumNumMiss, curRdl, prevRdl, minRdl = LLONG_MAX;
	for (int i = 0; i < numPLabeled; i++) {

//	    printf("%d %d >>>>\n", i, rankedInds[i]);

		ts = tss + rankedInds[i] * tsLen;

//		printf("@#####\n");

		discretize(hypoSeq, ts, tsLen, card);

//		printf("After discretization.\n");

		prevRdl = LLONG_MAX;
		cumNumMiss = 0;
		preNumP = 0;
		for (int j = numPLabeled; j < maxNumP; j++) {
//		    printf("%d %d\n", rankedInds[j], tsLen);
			ts = tss + rankedInds[j] * tsLen;
//			printf("Before\n");
			curRdl = getRdl(hypoSeq, cumNumMiss, ts, nextSeq, numTrain, j + 1, tsLen, card);
//			printf("curRdl = %d\n", curRdl);

			if (j < minNumP || curRdl < prevRdl) {
				prevRdl = curRdl;
			}
			else {
				preNumP = j;
				break;
			}
		}
		if (!preNumP)
			preNumP = maxNumP;
		if (prevRdl < minRdl) {
			minRdl = prevRdl;
			optPreNumP = preNumP;
		}
	}
//	printf("!!!\n");
	return optPreNumP;
}

//This implementation cannot handle cases where there are consecutive identical values in minNNDists correctly. However, we believe such cases are rare.
//Also, in this implementation, overlapping intervals are not allowed, except that the finishing point of one can be the starting point of the next.
extern "C" std::vector<std::tuple<int, int, int, int>> getIntervals(double* minNNDists, int numTrain, int numPLabeled, double beta) {

	std::vector<std::tuple<int, int, int, int>> intervals;	//start, ad, ds, finish
	int start, ad, ds, finish, instTrend, prevTrend, curTrend;
	prevTrend = 0; curTrend = -1; //1 for ascend, -1 for descend, 0 for stable
	double minNNDist, prevMinNNDist, diff, hd, lb, ub;
	prevMinNNDist = minNNDists[0];
	hd = INT_MAX; lb = INT_MAX; ub = INT_MIN;
	start = ad = ds = finish = INT_MAX;
	for (int i = 1; i < numTrain - numPLabeled; i++) {

		minNNDist = minNNDists[i];
		if (!doubleIsEqual(minNNDist, lb) && !doubleIsEqual(minNNDist, ub) &&
			minNNDist > lb && minNNDist < ub) {
			instTrend = 0;
		}
		else {
			diff = minNNDist - prevMinNNDist;
			instTrend = sign(diff);
			if (!instTrend) {
				instTrend = curTrend;
			}
		}

		if (curTrend != instTrend) {
			curTrend = instTrend;
		}

		if (prevTrend == 0) {
			if (curTrend == 0) {

				if (i == numTrain - numPLabeled - 1) {
					finish = i;
					if (start < ad && ad < ds && ds < finish) {
						intervals.push_back(std::make_tuple(start, ad, ds, finish));
					}
				}
			}
			else if (curTrend == 1) {
				finish = i - 1;
				if (start < ad && ad < ds && ds < finish) {
					intervals.push_back(std::make_tuple(start, ad, ds, finish));
				}
				start = i - 1;
				hd = INT_MAX; lb = INT_MAX; ub = INT_MIN;
			}
			else {
				finish = i - 1;
				if (start < ad && ad < ds && ds < finish) {
					intervals.push_back(std::make_tuple(start, ad, ds, finish));
				}
				hd = INT_MAX; lb = INT_MAX; ub = INT_MIN;
			}
		}
		else if (prevTrend == 1) {
			if (curTrend == -1) {
				ad = i - 1;
				hd = -diff;
			}
		}
		else {
			if (curTrend == 0) {

				//This case is impossible.
			}
			else if (curTrend == 1) {

				if (hd != INT_MAX) {
					lb = prevMinNNDist - beta * hd;
					ub = prevMinNNDist + beta * hd;
				}
				else {
					lb = INT_MAX; ub = INT_MIN;
				}
				if (!doubleIsEqual(minNNDist, lb) && !doubleIsEqual(minNNDist, ub) &&
					minNNDist > lb && minNNDist < ub) {
					curTrend = 0;
					ds = i - 1;

					if (i == numTrain - numPLabeled - 1) {
						finish = i;
						if (start < ad && ad < ds && ds < finish) {
							intervals.push_back(std::make_tuple(start, ad, ds, finish));
						}
					}

				}
				else {
					start = i - 1;
					hd = INT_MAX; lb = INT_MAX; ub = INT_MIN;
				}

			}
			else {
				if (hd != INT_MAX) {
					hd -= diff;
				}
			}
		}
		prevTrend = curTrend;
		prevMinNNDist = minNNDist;
	}
	return intervals;
}

//Mabel Gonz¨￠lez Castellanos, Christoph Bergmeir, Isaac Triguero, Yanet Rodr¨aguez, Jos¨| Manuel Ben¨atez: On the stopping criteria for k - Nearest Neighbor in positive unlabeled time series classification problems. INT_MAX.Sci. 328: 42-59 (2016)
extern "C" void sc_GBTRM(int* preNumPs, double* minNNDists, int minNumP, int maxNumP, int numTrain, int numPLabeled, double beta) {

	int initNumU = numTrain - numPLabeled;
	double max_minNNDists = max(minNNDists, initNumU);

	int start, finish, ad, ds, ip, ws, preNumP;
	double maxScs[5], scs[5], ha, hd, max_interval, lw;
	for (int i = 0; i < 5; i++)
		preNumPs[i] = maxScs[i] = INT_MIN;

	std::tuple<int, int, int, int> curInterval;
	std::vector<std::tuple<int, int, int, int>> intervals = getIntervals(minNNDists, numTrain, numPLabeled, beta);
	for (int i = 0; i < intervals.size(); i++) {

		curInterval = intervals[i];
		start = std::get<0>(curInterval);
		finish = std::get<3>(curInterval);
		ad = std::get<1>(curInterval);
		ds = std::get<2>(curInterval);

		ha = minNNDists[ad] - minNNDists[start];
		hd = minNNDists[ad] - minNNDists[ds];
		ws = finish - ds;
		max(max_interval, ip, minNNDists + start, finish - start + 1);
		ip += start;
		lw = (double)(initNumU - ip) / initNumU;

		scs[0] = hd * lw;
		scs[1] = ha * lw;
		scs[2] = ws * lw;
		scs[3] = max(ha, hd) * lw;
		scs[4] = max(hd / max_minNNDists, (double)ws / (initNumU - 1)) * lw;
		for (int j = 0; j < 5; j++) {
			if (scs[j] > maxScs[j]) {
				preNumP = numPLabeled + ip;
				if (preNumP >= minNumP && preNumP <= maxNumP) {
					maxScs[j] = scs[j];
					preNumPs[j] = preNumP;
				}
			}
		}
	}