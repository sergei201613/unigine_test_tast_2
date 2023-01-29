// this file you need to fill
// этот файл вам нужно заполнить
#include "task.h"

#include <cmath>
#include <thread>
#include <immintrin.h>

namespace KarelinSA
{
#ifdef _MSC_VER
#  define __builtin_popcount __popcnt
#endif

	const float PI = 3.1415926535f;
	const float PI_HALF = PI / 2.0f;
	const float PI_QUART = PI / 4.0f;
	const float RAD2DEG = 180.0f / PI;
	const float DEG2RAD = PI / 180.0f;
	const float DEG2RAD_HALF = PI / 180.0f / 2.0f;

	inline vec2 vec2Sub(const vec2 a, const vec2 b)
	{
		vec2 res;
		res.x = a.x - b.x;
		res.y = a.y - b.y;
		return res;
	}

	inline float vec2DistSqr(const vec2 a, const vec2 b)
	{
		float ax_sub_bx = a.x - b.x;
		float ay_sub_by = a.y - b.y;
		return ax_sub_bx * ax_sub_bx + ay_sub_by * ay_sub_by;
	}

	inline float vec2Dot(const vec2 a, const vec2 b)
	{
		return a.x * b.x + a.y * b.y;
	}

	inline float vec2Cross(const vec2 a, const vec2 b)
	{
		return a.y * b.x - a.x * b.y;
	}

	inline float vec2Len(const vec2 a)
	{
		return std::sqrt(a.x * a.x + a.y * a.y);
	}

	inline float vec2LenSqr(const vec2 a)
	{
		return a.x * a.x + a.y * a.y;
	}

	inline vec2 vec2Div(const vec2 v, const float s)
	{
		vec2 res;
		res.x = v.x / s;
		res.y = v.y / s;
		return res;
	}

	inline vec2 vec2Unit(const vec2 a)
	{
		return vec2Div(a, vec2Len(a));
	}

	inline float cosFast(float a)
	{
		float a2 = a * a;
		return (((((-2.605e-07f * a2 + 2.47609e-05f) * a2 - 0.0013888397f) * a2 + 
			0.0416666418f) * a2 - 0.4999999963f) * a2 + 1.0f);
	}

	void handleUnits(const std::vector<unit>& units, std::vector<int>& result, 
		const int start, const int end)
	{
		int unit_count = units.size();
		int tail_size = unit_count % 8;
		int main_size = unit_count - tail_size;

		float* main_positions_x = new float[main_size];
		float* main_positions_y = new float[main_size];

		for (int i = 0; i < main_size; ++i)
		{
			main_positions_x[i] = units[i].position.x;
			main_positions_y[i] = units[i].position.y;
		}

		for (int i = start; i < end; ++i)
		{
			unit a = units[i];
			vec2 a_dir = a.direction;

			int count = 0;

			float a_fov_half_dot = cosFast(a.fov_deg * DEG2RAD_HALF);
			float a_dist_sqr = a.distance * a.distance;

			__m256 a_pos_x_ = _mm256_set1_ps(a.position.x);
			__m256 a_pos_y_ = _mm256_set1_ps(a.position.y);

			__m256 a_dist_sqr_ = _mm256_set1_ps(a_dist_sqr);

			__m256 a_dir_x_ = _mm256_set1_ps(a.direction.x);
			__m256 a_dir_y_ = _mm256_set1_ps(a.direction.y);

			__m256 a_fov_half_dot_ = _mm256_set1_ps(a_fov_half_dot);

			// vector processing
			for (int j = 0; j < main_size; j += 8)
			{
				__m256 ab_dir_x_ = _mm256_sub_ps(
					_mm256_loadu_ps(&main_positions_x[j]),
					a_pos_x_
				);

				__m256 ab_dir_y_ = _mm256_sub_ps(
					_mm256_loadu_ps(&main_positions_y[j]),
					a_pos_y_
				);

				__m256 dist_sqr_ = _mm256_add_ps(
					_mm256_mul_ps(ab_dir_x_, ab_dir_x_),
					_mm256_mul_ps(ab_dir_y_, ab_dir_y_)
				);

				__m256 dist_mask = _mm256_cmp_ps(dist_sqr_, a_dist_sqr_, 
					_CMP_LT_OQ);

				__m256 dir_diff_dot_ = _mm256_add_ps(
					_mm256_mul_ps(a_dir_x_, ab_dir_x_),
					_mm256_mul_ps(a_dir_y_, ab_dir_y_)
				);

				__m256 angle_mask = _mm256_cmp_ps(
					dir_diff_dot_,
					_mm256_mul_ps(
						a_fov_half_dot_,
						_mm256_sqrt_ps(dist_sqr_)
					),
					_CMP_GT_OQ
				);

				 count += __builtin_popcount(
					 _mm256_movemask_ps(
						 _mm256_and_ps(dist_mask, angle_mask)
					 )
				 );
			}

			// scalar processing for remaining units
			for (int j = main_size; j < unit_count; ++j)
			{
				vec2 ab_dir
				{
					units[j].position.x - a.position.x,
					units[j].position.y - a.position.y
				};

				float dist_sqr = ab_dir.x * ab_dir.x + ab_dir.y * ab_dir.y;

				if (dist_sqr > a_dist_sqr)
					continue;

				float dir_diff_dot = a_dir.x * ab_dir.x + a_dir.y * ab_dir.y;

				if (dir_diff_dot > a_fov_half_dot * std::sqrt(dist_sqr))
					++count;
			}

			result[i] = count;
		}

		delete[] main_positions_x;
		delete[] main_positions_y;
	}

	void handleUnitsAsync(const std::vector<unit>& units, const int unit_count,
		std::vector<int>& result)
	{
		int thread_count = std::thread::hardware_concurrency();

		// chunk count can't be more than unit count
		int chunk_count = std::min(thread_count, unit_count);

		std::thread* threads = new std::thread[thread_count];

		int chunk_size = unit_count / chunk_count;
		int unit_remains = unit_count % chunk_count;
		int chunk_start = 0;

		for (int i = 0; i < chunk_count; ++i)
		{
			int chunk_end = chunk_start + chunk_size;

			if (unit_remains != 0)
			{
				chunk_end++;
				unit_remains--;
			}

			threads[i] = std::thread(handleUnits, std::ref(units),
				std::ref(result), chunk_start, chunk_end);

			chunk_start = chunk_end;
		}

		for (int i = 0; i < chunk_count; i++)
			threads[i].join();

		delete[] threads;
	}
}

void Task::checkVisible(const std::vector<unit> &input_units, 
	std::vector<int> &result)
{
	int size = input_units.size();
	result = std::vector<int>(size);
	KarelinSA::handleUnitsAsync(input_units, size, result);
}
