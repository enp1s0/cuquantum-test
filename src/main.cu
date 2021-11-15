#include <cuComplex.h>
#include <custatevec.h>
#include <iostream>

using statevector_t = cuDoubleComplex;

template <class T>
struct get_real;
template <> struct get_real<cuDoubleComplex> {using type = double;};
template <> struct get_real<cuComplex> {using type = float;};

template <class T>
cudaDataType_t get_data_type();
template <> cudaDataType_t get_data_type<cuComplex      >() {return CUDA_C_32F;}
template <> cudaDataType_t get_data_type<cuDoubleComplex>() {return CUDA_C_64F;}

template <class T>
custatevecComputeType_t get_custatevec_compute_type();
template <> custatevecComputeType_t  get_custatevec_compute_type<cuComplex      >() {return CUSTATEVEC_COMPUTE_32F;}
template <> custatevecComputeType_t  get_custatevec_compute_type<cuDoubleComplex>() {return CUSTATEVEC_COMPUTE_64F;}

void check_custatevec_error(
		const custatevecStatus_t status,
		const char* const filepath,
		const unsigned line,
		const char* const funcname
		) {
	if (status != CUSTATEVEC_STATUS_SUCCESS) {
		auto get_str = [](const custatevecStatus_t state) {
			switch(state) {
#define CUSTATEVEC_ERROR_CASE_MESSAGE(x) case x: return #x
				CUSTATEVEC_ERROR_CASE_MESSAGE(CUSTATEVEC_STATUS_SUCCESS                 );
				CUSTATEVEC_ERROR_CASE_MESSAGE(CUSTATEVEC_STATUS_NOT_INITIALIZED         );
				CUSTATEVEC_ERROR_CASE_MESSAGE(CUSTATEVEC_STATUS_ALLOC_FAILED            );
				CUSTATEVEC_ERROR_CASE_MESSAGE(CUSTATEVEC_STATUS_INVALID_VALUE           );
				CUSTATEVEC_ERROR_CASE_MESSAGE(CUSTATEVEC_STATUS_ARCH_MISMATCH           );
				CUSTATEVEC_ERROR_CASE_MESSAGE(CUSTATEVEC_STATUS_EXECUTION_FAILED        );
				CUSTATEVEC_ERROR_CASE_MESSAGE(CUSTATEVEC_STATUS_INTERNAL_ERROR          );
				CUSTATEVEC_ERROR_CASE_MESSAGE(CUSTATEVEC_STATUS_NOT_SUPPORTED           );
				CUSTATEVEC_ERROR_CASE_MESSAGE(CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE  );
				CUSTATEVEC_ERROR_CASE_MESSAGE(CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED);
			default:
				return "Unknown";
			}
		};
		std::printf("ERROR: %s @ %s, line %u (%s)\n", get_str(status), filepath, line, funcname);
		exit(1);
	}
}

#define CHECH_CUSTATEVEC_ERROR(status) check_custatevec_error((status), __FILE__, __LINE__, __func__)

void init_statevector(statevector_t* const ptr,
		const std::size_t statevector_length) {
	cudaMemset(ptr, 0, sizeof(statevector_t) * statevector_length / sizeof(int));

	statevector_t zero;
	zero.x = 1;
	zero.y = 0;

	cudaMemcpy(ptr, &zero, sizeof(zero), cudaMemcpyDefault);
}

void gate_H(custatevecHandle_t handle,
		statevector_t* const ptr,
		const unsigned target_qubit,
		const unsigned num_qubits) {
	const auto sqrt2 = std::sqrt(2.);
	constexpr unsigned adjoint = 0;
	statevector_t matrix[4];
	for (unsigned i = 0; i < 4; i++) {
		matrix[i].x = 1 / sqrt2;
		matrix[i].y = 1 / sqrt2;
	}
	matrix[3].x *= -1;
	matrix[3].y *= -1;

	std::printf("[%10s] custatevecApplyMatrix_bufferSize start\n", __func__);
	void* working_memory;
	std::size_t working_memory_size;
	CHECH_CUSTATEVEC_ERROR(custatevecApplyMatrix_bufferSize(
			handle,
			get_data_type<statevector_t>(),
			num_qubits,
			matrix,
			get_data_type<statevector_t>(),
			CUSTATEVEC_MATRIX_LAYOUT_COL,
			adjoint,
			1,
			0,
			get_custatevec_compute_type<statevector_t>(),
			&working_memory_size
			));

	if (working_memory_size) {
		cudaMalloc(&working_memory, working_memory_size);
	}
	std::printf("[%10s] cudaMalloc start (%lu B)\n", __func__, working_memory_size);

	std::printf("[%10s] custatevecApplyMatrix start\n", __func__);
	int targets[] = {(int)target_qubit};
	int controls[] = {};
	CHECH_CUSTATEVEC_ERROR(custatevecApplyMatrix(
			handle,
			ptr,
			get_data_type<statevector_t>(),
			num_qubits,
			matrix,
			get_data_type<statevector_t>(),
			CUSTATEVEC_MATRIX_LAYOUT_COL,
			adjoint,
			targets,
			1,
			controls,
			0,
			nullptr,
			get_custatevec_compute_type<statevector_t>(),
			working_memory,
			working_memory_size
			));

	cudaFree(working_memory);
}

int main() {
	constexpr unsigned num_qubits = 29;
	constexpr std::size_t statevector_length = 1lu << num_qubits;

	statevector_t *statevector;
	cudaMalloc(&statevector, sizeof(statevector_t) * statevector_length);

	init_statevector(statevector, statevector_length);

	std::printf("[%10s] custatevecCreate start\n", __func__);
	custatevecHandle_t handle;
	CHECH_CUSTATEVEC_ERROR(custatevecCreate(&handle));

	for (unsigned i = 0; i < num_qubits; i++) {
		gate_H(handle, statevector, i, num_qubits);
	}

	std::printf("[%10s] custatevecDestroy start\n", __func__);
	CHECH_CUSTATEVEC_ERROR(custatevecDestroy(handle));

	cudaFree(statevector);

	return 0;
}
