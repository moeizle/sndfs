#ifndef __CLPP_STOPWATCH_H__
#define __CLPP_STOPWATCH_H__


#include <windows.h>

typedef struct
{
	LARGE_INTEGER start;
	LARGE_INTEGER stop;
} stopWatch;




	class StopWatch
	{
	private:
		stopWatch timer;
		LARGE_INTEGER frequency;
		double LIToSecs(LARGE_INTEGER& L);

	public:
		StopWatch();

		void StartTimer();
		void StopTimer();

		double GetElapsedTime();
	};


#endif