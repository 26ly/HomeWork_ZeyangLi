#include <iostream>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <cmath>

using namespace std;

class Matrix {
private:
	vector<vector<double>> data;
	size_t rows;
	size_t cols;
public:
	// 初始化列表构造函数
	Matrix(initializer_list<initializer_list<double>> ori_matrix)
		: rows(ori_matrix.size()), cols(!ori_matrix.size() ? 0 : ori_matrix.begin() -> size()) {
		// 初始化data的大小
		data.resize(rows);                    // 先设置行数
		for (size_t k = 0; k < rows; k++) {
			data[k].resize(cols);             // 再设置每行的列数, k为行索引变量
		}

		size_t k = 0;
		for (const auto& row : ori_matrix) {
			if (row.size() != cols) {
				throw invalid_argument("某矩阵第 " + to_string(k + 1) + " 行与首行列数不一致");
			}
			k++;
		}
		size_t i = 0;
		for (const auto& row : ori_matrix) {  // row 是 initializer_list<double>，即一层解引用
			size_t j = 0;
			for (double value : row) {        // value 是 double
				data[i][j] = value;
				j++;
			}
			i++;
		}
	}

	// 零矩阵构造函数
	explicit Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
		data.resize(rows, vector<double>(cols, 0.0));
	}

	// 深拷贝构造函数
	Matrix(const Matrix& copy) : data(copy.data), rows(copy.rows), cols(copy.cols) {}

	// 获取行数
	size_t GetRows() const { return rows; }

	// 获取列数
	size_t GetCols() const { return cols; }

	// 矩阵某一单值赋值
	void Assignment(size_t i, size_t j, double assign) {
		if (i >= rows || j >= cols) {
			throw invalid_argument("赋值索引越界！");
		}
		data[i][j] = assign;
	}

	// 获取矩阵某一单值
	double operator()(size_t i, size_t j) const {
		return data[i][j];
	}

	// 矩阵加法
	Matrix operator+(const Matrix& other) const {
		if (other.rows != rows || other.cols != cols) {
			throw invalid_argument("两矩阵维度不同，无法相加");
		}
		Matrix result(rows, cols);
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				result.Assignment(i, j, data[i][j] + other(i, j));
			}
		}
		return result;
	}

	// 矩阵减法
	Matrix operator-(const Matrix& other) const {
		if (other.rows != rows || other.cols != cols) {
			throw invalid_argument("两矩阵维度不同，无法相减");
		}
		Matrix result(rows, cols);
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				result.Assignment(i, j, data[i][j] - other(i, j));
			}
		}
		return result;
	}

	// 矩阵乘法
	Matrix operator*(const Matrix& other) const {
		if (other.rows != cols) {
			throw invalid_argument("乘矩阵列数与被乘矩阵行数不等，无法相乘");
		}
		Matrix result(rows, other.cols);
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < other.cols; j++) {
				for (size_t k = 0; k < other.rows; k++) {
					result.Assignment(i, j, result(i, j) + data[i][k] * other(k, j));
				}
			}
		}
		return result;
	}

	// 矩阵数乘（右乘）
	Matrix operator*(const double a) const {
		Matrix result(rows, cols);
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				result.Assignment(i, j, a * data[i][j]);
			}
		}
		return result;
	}

	// 矩阵数乘（左乘）
	friend Matrix operator*(const double a, const Matrix& other) {
		return other * a;
	}

	// 矩阵转置
	Matrix transpose() const {
		Matrix result(cols, rows);
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				result.Assignment(j, i, data[i][j]);
			}
		}
		return result;
	}

	// 同维度矩阵赋值
	Matrix& operator=(const Matrix& other) {
		// 自赋值检查
		if (this == &other) {
			return *this;
		}
		// 正常赋值
		if (other.rows != rows || other.cols != cols) {
			throw invalid_argument("两矩阵维度不同，无法赋值");
		}
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				data[i][j] = other(i, j);
			}
		}
		return *this;
	}

	// 矩阵转向量函数
	static std::vector<double> ToVector(Matrix matrix) {
		if (matrix.cols != 1) {
			throw invalid_argument("该矩阵列数不为1，无法转化为向量，请转化成\"n*1\"形式");
		}
		std::vector<double> a;
		a.resize(matrix.rows);
		for (size_t i = 0; i < matrix.rows; i++) {
			a[i] = matrix(i, 0);
		}
		return a;
	}

	// 向量转矩阵函数
	static Matrix ToMatrix(vector<double> a) {
		Matrix result(a.size(), 1);  // 创建 n*1 的矩阵
		for (size_t i = 0; i < a.size(); i++) {
			result.Assignment(i, 0, a[i]);  // 将向量元素放入第一列
		}
		return result;
	}

	// 矩阵输出可视化
	friend ostream& operator<<(ostream& os, const Matrix& matrix) {
		for (size_t i = 0; i < matrix.rows; i++) {
			for (size_t j = 0; j < matrix.cols; j++) {
				os << setw(10) << fixed << setprecision(2) << matrix(i, j);
			}
			os << endl;
		}
		return os;
	}

	// 析构函数
	~Matrix() {}
};

class MatrixProcesser {
private:
public:
	//第一种向量空间变换函数，实现变换过程可视化
	static vector<double> TransformVectorV1() {
		try {
			double x;
			double y;
			double dx;
			double dy;
			double theta;
			Matrix Z_0(2, 1);
			Matrix Z_r(2, 1);
			Matrix Z_t(2, 1);
			Matrix rotation(2, 2);
			Matrix move(2, 1);
			//先旋转，后平移
			cout << endl << endl << "请输入空间坐标P(x,y)" << endl << "x坐标：";
			if (!(cin >> x)) {
				throw invalid_argument("输入错误：x坐标必须是整数！");
			}
			cout << "y坐标：";
			if (!(cin >> y)) {
				throw invalid_argument("输入错误：y坐标必须是整数！");
			}
			Z_0 = { {x},{y} };
			cout << "请输入旋转角（弧度）：";
			if (!(cin >> theta)) {
				throw invalid_argument("输入错误：旋转角必须是整数！");
			}
			rotation = { {cos(theta),-sin(theta)},
						 {sin(theta),cos(theta)} };
			Z_r = rotation * Z_0;
			//输出旋转后，平移前坐标
			cout << "旋转后坐标为：" << endl << Z_r << endl << "请输入平移坐标" << endl;
			cout << "x坐标：";
			if (!(cin >> dx)) {
				throw invalid_argument("输入错误：平移x坐标必须是整数！");
			}
			cout << "y坐标：";
			if (!(cin >> dy)) {
				throw invalid_argument("输入错误：平移y坐标必须是整数！");
			}
			move = { {dx},{dy} };
			Z_t = Z_r + move;
			//输出最后坐标
			cout << "旋转平移后坐标为：" << endl << Z_t << endl;
			return Matrix::ToVector(Z_t);
		}
		catch (const exception& e) {
			cout << e.what() << endl;
		}
	}

	//第二种向量空间变换函数，用扩增矩阵法实现端到端
	//参数说明：初始x，初始y，旋转角，平移x，平移y
	static vector<double> TransformVectorV2(double x, double y, double theta, double dx, double dy) {
		Matrix turn_mat = { {cos(theta),-sin(theta),dx},
							{sin(theta),cos(theta) ,dy},
							{0		   ,0		   ,1 } };
		Matrix Z_0 = { {x},{y},{1} };
		Matrix Z_t = turn_mat * Z_0;
		std::vector<double> a = Matrix::ToVector(Z_t);
		a.resize(2);
		return a;
	}
};

//类的使用方法参照下面案例
int main() {
	try {
		Matrix A0(5, 6);
		Matrix A = { {5,5,7.146},
					 {2,9,3},
					 {10,4,6} };
		Matrix B = { {0.55,4.5,3.1},
					 {1,29,-3},
					 {-10.5,4.9,6.2} };
		Matrix C = A * B;
		Matrix D = A + B;
		Matrix E = A - B;

		cout << "A0 = " << endl << A0 << endl
			<< "A = " << endl << A << endl
			<< "B = " << endl << B << endl
			<< "C = " << endl << C << endl
			<< "D = " << endl << D << endl
			<< "E = " << endl << E << endl;
		// 单值修改
		A.Assignment(2, 1, 114.514);			//<——2和3是数组索引，即矩阵第3行第2列
		cout << "A_new = " << endl << A << endl;
		// 矩阵整体赋值
		C = A;
		cout << "C_new = " << endl << C << endl;
		//测试V1函数
		MatrixProcesser::TransformVectorV1();
		//册数V2函数
		vector<double> a = MatrixProcesser::TransformVectorV2(1, 2, 1.57, 2, 3);
		//下面两行只是将V2函数输出的vector类型向量转换为矩阵表示，由于vector类本身未定义cout输出重载
		Matrix output = Matrix::ToMatrix(a);
		cout << "output = " << endl << output << endl;
	}
	catch (const invalid_argument& e) {
		cout << e.what() << endl;
	}
	return 0;
}
