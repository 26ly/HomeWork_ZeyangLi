#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
using namespace std;
using namespace cv;

// ×Ô¶¨ÒåÒì³£Àà
class ImageException : public std::exception {
private:
	std::string message;
public:
	ImageException(const std::string& msg) : message(msg) {}
	//Ğéº¯ÊıÖØĞ´
	const char* what() const noexcept override {
		return message.c_str();
	}
};

class ImageProcesser {
private:
	string ImgPath;
	cv::Mat Img;
	bool isLoaded;
	// ¼ì²éÍ¼ÏñÊÇ·ñ³É¹¦¼ÓÔØ
	void CheckImageLoaded() const {
		if (!isLoaded || Img.empty()) {
			throw ImageException("Í¼ÏñÎ´³É¹¦¼ÓÔØ£¬ÎŞ·¨Ö´ĞĞ²Ù×÷£¬ÇëÏÈÀûÓÃ\"ImageProcesser(PATH)\"½øĞĞ¼ÓÔØ");
		}
	}
public:
	// ¹¹Ôìº¯Êı
	ImageProcesser(const string& filename) : ImgPath(filename), isLoaded(false) {
		Img = cv::imread(ImgPath, 1);
		//Â·¾¶¸ñÊ½Òì³£ÅĞ¶Ï
		if (Img.empty()) {
			throw ImageException("ÎŞ·¨¼ÓÔØÍ¼ÏñÎÄ¼ş: " + ImgPath + " - Çë¼ì²éÎÄ¼şÂ·¾¶ºÍ¸ñÊ½ÊÇ·ñÖ§³Ö");
		}
		isLoaded = true;
		cout << "Í¼Ïñ¼ÓÔØ³É¹¦: " << ImgPath << endl;
	}

	// »ñÈ¡Í¼Ïñ³ß´ç
	vector<int> GetSize() const {
		CheckImageLoaded();
		return { Img.cols,Img.rows };	// ¿í¶È*¸ß¶È¼´ÁĞ*ĞĞ
	}

	// »ñÈ¡Í¼ÏñÍ¨µÀÊı
	int GetChannels() const {
		CheckImageLoaded();
		return Img.channels();
	}

	// »ñÈ¡Í¼ÏñÏñËØÊı¾İ£¨·µ»ØMatÒıÓÃ£©
	const cv::Mat& GetPixelData() const {
		CheckImageLoaded();
		return Img;
	}

	// ÏÔÊ¾Í¼Ïñ
	void ShowImage(const string& windowName = "Image") const {
		char a;
		CheckImageLoaded();
		Mat CopyImg = Img.clone();
		putText(CopyImg, windowName, Point(10, 30), 3, 1, Scalar(0, 200, 190), 2);
		// putText²ÎÊı£º´°¿Ú£¬´òÓ¡ÄÚÈİ£¬Î»ÖÃ£¬×ÖÌå£¬×ÖºÅ£¬»Ò¶È£¬´ÖÏ¸
		cv::imshow(windowName, CopyImg);
		// ÓÃ»§ÊäÈë¹Ø±Õ´°¿Ú
		cout << "°´ÈÎÒâ¼ü¹Ø±Õ±¾´°¿Ú£¬»ò°´ESCÍË³ö½ø³Ì..." << endl;
		int key = cv::waitKey(0);  // µÈ´ı°´¼ü
		// ¼ì²éÊÇ·ñ°´ÁËESC¼ü
		if (key == 27) {
			exit(0);
		}
	}

	// »ù´¡ÈÎÎñ¹¦ÄÜ
	// 1. RGB×ª»Ò¶ÈÍ¼
	cv::Mat ToGray() const {
		CheckImageLoaded();
		if (Img.channels() != 3) {
			throw ImageException("µ±Ç°Í¼Ïñ²»ÊÇRGB¸ñÊ½£¬ÎŞ·¨×ª»»Îª»Ò¶ÈÍ¼");
		}

		cv::Mat GrayImg;
		cv::cvtColor(Img, GrayImg, COLOR_BGR2GRAY);
		cout << "ÒÑ×ª»»Îª»Ò¶ÈÍ¼" << endl;
		return GrayImg;
	}

	// 2. ¾ùÖµÄ£ºıÈ¥Ôë
	cv::Mat ToMeanBlur(int kernelSize = 5) const {
		CheckImageLoaded();
		if (kernelSize % 2 == 0) {
			throw ImageException("¾í»ıºË´óĞ¡±ØĞëÎªÆæÊı");
		}

		cv::Mat BlurredImg;
		cv::blur(Img, BlurredImg, Size(kernelSize, kernelSize));
		cout << "ÒÑÓ¦ÓÃ¾ùÖµÄ£ºı£¬¾í»ıºË´óĞ¡: " << kernelSize << "*" << kernelSize << endl;

		// ¾ùÖµ½µÔë¿ÉÊÓ»¯
		Mat CopyImg = BlurredImg.clone();
		putText(CopyImg, "BlurredImg", Point(10, 30), 3, 1, Scalar(255), 2);
		imshow("BlurredImg", CopyImg);
		cout << "°´ÈÎÒâ¼ü¼ÌĞø£¬»ò°´ESCÍË³ö½ø³Ì..." << endl;
		int key = cv::waitKey(0);  // µÈ´ı°´¼ü
		// ¼ì²éÊÇ·ñ°´ÁËESC¼ü
		if (key == 27) {
			exit(0);
		}

		return BlurredImg;
	}

	// 3. ¸ßË¹Ä£ºı
	cv::Mat ToGaussianBlur(int kernelSize = 5, double sigma = 1.0) const {
		CheckImageLoaded();
		if (kernelSize % 2 == 0) {
			throw ImageException("¾í»ıºË´óĞ¡±ØĞëÎªÆæÊı");
		}

		cv::Mat GaussianImg;
		cv::GaussianBlur(Img, GaussianImg, Size(kernelSize, kernelSize), sigma);
		cout << "ÒÑÓ¦ÓÃ¸ßË¹Ä£ºı£¬¾í»ıºË´óĞ¡: " << kernelSize << "x" << kernelSize << endl;

		// ¸ßË¹Ä£ºı¿ÉÊÓ»¯
		Mat CopyImg = GaussianImg.clone();
		putText(CopyImg, "GaussianImg", Point(10, 30), 3, 1, Scalar(255), 2);
		imshow("GaussianImg", CopyImg);
		cout << "°´ÈÎÒâ¼ü¼ÌĞø£¬»ò°´ESCÍË³ö½ø³Ì..." << endl;
		int key = cv::waitKey(0);  // µÈ´ı°´¼ü
		// ¼ì²éÊÇ·ñ°´ÁËESC¼ü
		if (key == 27) {
			exit(0);
		}

		return GaussianImg;
	}

	// 4. Í¼Ïñ¶şÖµ»¯
	// ¾­¹ı·´¸´ÊµÑé£¬È¡105~210Ê±£¬µÆÌõÏà¶ÔÇåÎú£¬¶øÂÖÌ¥·´¹âÎó²îÏà¶Ô½ÏĞ¡£¬Îª×î¼Ñ·¶Î§£¬¹ÊÉèÄ¬ÈÏÖµ
	cv::Mat ToBinary(double MinValue = 105, double MaxValue = 210) const {
		CheckImageLoaded();
		cv::Mat GrayImg = ToGray();		// ×ª»Ò¶È
		cv::Mat turnImg;				// ÖĞ¼äÌ¬
		cv::Mat BinaryImg;

		// ÏÈ¶şÖµ»¯È·¶¨ÏÂÏŞ
		cv::threshold(GrayImg, turnImg, MaxValue, 255, THRESH_TOZERO_INV);
		// ÔÙ·´¶şÖµ»¯È¡ÁãÈ·¶¨ÉÏÏŞ
		cv::threshold(turnImg, BinaryImg, MinValue, 255, THRESH_TOZERO);
		cout << "ÒÑÓ¦ÓÃ¶şÖµ»¯£¬" << MinValue << " < ãĞÖµ < " << MaxValue << endl;

		//// ¶şÖµ»¯Í¼Æ¬¿ÉÊÓ»¯
		//Mat CopyImg = BinaryImg.clone();
		//putText(CopyImg, "BinaryImg", Point(10, 30), 3, 1, Scalar(255), 2);
		//imshow("BinaryImg", CopyImg);
		//cout << "°´ÈÎÒâ¼ü¼ÌĞø£¬»ò°´ESCÍË³ö½ø³Ì..." << endl;
		//int key = cv::waitKey(0);  // µÈ´ı°´¼ü
		//// ¼ì²éÊÇ·ñ°´ÁËESC¼ü
		//if (key == 27) {
		//	exit(0);
		//}
		return BinaryImg;
	}

	// func:¸³ÖµÔËËã
	ImageProcesser& operator=(const ImageProcesser& other) {
		if (this != &other) {
			ImgPath = other.ImgPath;
			Img = other.Img.clone();
			isLoaded = other.isLoaded;
		}
		return *this;
	}

	// func:±£´æ´¦ÀíºóµÄÍ¼Ïñ
	void saveImage(const string& oPath) const {
		CheckImageLoaded();
		if (!cv::imwrite(oPath, Img)) {
			throw ImageException("±£´æÍ¼ÏñÊ§°Ü: " + oPath);
		}
		cout << "Í¼ÏñÒÑ±£´æÓÚ: " << oPath << endl;
	}

	// struct:µÆÌõ¼ì²âºÍÉ¸Ñ¡¹¦ÄÜ
	struct LightBar {
		RotatedRect rect;			// Íâ½ÓĞı×ª¾ØĞÎ
		double WHR;					// ¿í¸ß±È(Width and Height Ratio)
		double area;				// Ãæ»ı
		vector<Point> contour;		// ÂÖÀªµã
	};

	// ±¾²¿·Ö¼û±¨¸æ
	// func:HSVÑÕÉ«¿Õ¼ä×ª»»
	cv::Mat ToHSV() const {

		CheckImageLoaded();
		if (Img.channels() != 3) {
			throw ImageException("µ±Ç°Í¼Ïñ²»ÊÇRGB¸ñÊ½£¬ÎŞ·¨×ª»»ÎªHSV");
		}

		cv::Mat HSVImg;
		cv::cvtColor(Img, HSVImg, COLOR_BGR2HSV);
		cout << "ÒÑ×ª»»ÎªHSVÑÕÉ«¿Õ¼ä" << endl;
		return HSVImg;
	}

	// func:»ùÓÚHSVµÄãĞÖµ·Ö¸î
	cv::Mat ToHSVThreshold(double minH = 0, double maxH = 180,
		double minS = 0, double maxS = 255,
		double minV = 0, double maxV = 255) const {

		CheckImageLoaded();

		cv::Mat HSVImg = ToHSV();
		cv::Mat mask;
		cv::Mat mask1;
		cv::Mat mask2;

		// ´´½¨HSV·¶Î§ÑÚÂë£¨ÑÓÍØºìÉ«Çø¼ä£©
		if (minH < 0) {
			double irminH = 180 + minH;
			cv::inRange(HSVImg,
				Scalar(0, minS, minV),
				Scalar(maxH, maxS, maxV),
				mask1);
			cv::inRange(HSVImg,
				Scalar(irminH, minS, minV),
				Scalar(180, maxS, maxV),
				mask2);
			mask = mask1 | mask2;
		}
		else {
			cv::inRange(HSVImg,
				Scalar(minH, minS, minV),
				Scalar(maxH, maxS, maxV),
				mask);
		}
		cout << "ÒÑÓ¦ÓÃHSVãĞÖµ·Ö¸î: H[" << minH << "," << maxH
			<< "] S[" << minS << "," << maxS
			<< "] V[" << minV << "," << maxV << "]" << endl;

		return mask;
	}

	// func:»ùÓÚHSVµÄµÆÌõ¼ì²â£¨ºìÉ«µÆÌõ×¨ÓÃ£©
	vector<LightBar> SelectLightBarHSV(double minArea = 20, double maxArea = 5000,
		double minWHR = 1.2, double maxWHR = 30.0, double minH = -20.0, double maxH = 10.0,
		double minS = 100, double maxS = 255, double minV = 100, double maxV = 255) const {

		CheckImageLoaded();

		// æ³¨é‡Šæ‰æ—¥å¿—è®¾ç½®
		cout << ">>> ¿ªÊ¼HSVµÆÌõ¼ì²â <<<" << endl;

		// ºìÉ«ÔÚHSVÖĞÓĞÁ½¸ö·¶Î§£¨ÒòÎªºìÉ«ÔÚÉ«Ïà»·µÄÁ½¶Ë£©
		cv::Mat HSVImg = ToHSV();
		cv::Mat redMask;

		// ºìÉ«·¶Î§
		redMask = ToHSVThreshold(minH, maxH, minS, maxS, minV, maxV);

		vector<LightBar> lightBars;
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		findContours(redMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		cout << "ÕÒµ½ " << contours.size() << " ¸öºìÉ«ÂÖÀª" << endl;

		for (const auto& contour : contours) {
			double area = contourArea(contour);
			if (area < minArea || area > maxArea) {
				continue;
			}

			RotatedRect rect = minAreaRect(contour);
			double WHR = max(rect.size.width, rect.size.height) / min(rect.size.width, rect.size.height);

			if (WHR < minWHR || WHR > maxWHR) {
				continue;
			}

			LightBar lightBar;
			lightBar.rect = rect;
			lightBar.WHR = WHR;
			lightBar.area = area;
			lightBar.contour = contour;

			lightBars.push_back(lightBar);
		}
		cout << "ÆäÖĞ°üº¬ " << lightBars.size() << " ¸öÒÉËÆºìÉ«µÆÌõÂÖÀª" << endl;
		return lightBars;
	}

	// func:ÕÒ³öµÆÌõ±ß½ç
	vector<LightBar> SelectLightBar(double minArea = 20, double maxArea = 5000,
		double minWHR = 2.0, double maxWHR = 30.0) const {

		CheckImageLoaded();

		// ÔÚ³ÌĞò¿ªÊ¼´¦ÉèÖÃÈÕÖ¾¼¶±ğ - Ö»ÏÔÊ¾´íÎóĞÅÏ¢£¬²»´òÓ¡INFO¶àÓàĞÅÏ¢
		// æ³¨é‡Šæ‰æ—¥å¿—è®¾ç½®
		cout << ">>> ¿ªÊ¼µÆÌõ¼ì²â <<<" << endl;


		Mat binaryImg = ToBinary();
		//´´½¨½á¹¹ÌåÔªËØÁĞ±í£¬°üÀ¨Ã¿¸öµÆÌõµÄÏà¹ØĞÅÏ¢
		vector<LightBar> lightBars;
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(binaryImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		cout << "ÕÒµ½ " << contours.size() << " ¸öÂÖÀª" << endl;

		for (const auto& contour : contours) {
			// ÅĞ¶ÏÃæ»ıÊÇ·ñºÏÊÊ£¬¹ı´ó¹ıĞ¡Ôò¿ÉÄÜÎª±³¾°»òÒ»Ğ©ÔëÉù£¬´ËÊ±Ö±½ÓÌø¹ı²»½øĞĞºóĞø
			double area = contourArea(contour);
			if (area < minArea || area > maxArea) {
				continue;
			}
			// »ñÈ¡×îĞ¡Íâ½Ó¾ØĞÎ
			RotatedRect rect = minAreaRect(contour);
			// ¼ÆËã¿í¸ß±È£¨È·±£¿í > ¸ß£©
			double WHR = max(rect.size.width, rect.size.height) / min(rect.size.width, rect.size.height);
			// Í¬ÉÏ£¬ÅĞ¶Ï¸ß¿í±ÈÊÇ·ñºÏÊÊ£¬½µÔë
			if (WHR < minWHR || WHR > maxWHR) {
				continue;
			}
			// Ìí¼Óµ½½á¹û
			LightBar lightBar;
			lightBar.rect = rect;
			lightBar.WHR = WHR;
			lightBar.area = area;
			lightBar.contour = contour;

			lightBars.push_back(lightBar);
		}
		cout << "ÆäÖĞ°üº¬ " << lightBars.size() << " ¸öÒÉËÆµÆÌõÂÖÀª" << endl;
		return lightBars;
	}

	// func:ÔÚÔ­Ê¼Í¼ÏñÉÏ±ê¼ÇµÆÌõ
	Mat MarkLightBar(const vector<LightBar>& lightBars,
		bool drawContour = true,
		bool showInfo = true) const {

		CheckImageLoaded();

		Mat SignImg = Img.clone();

		for (size_t i = 0; i < lightBars.size(); i++) {
			const LightBar& bar = lightBars[i];

			// »æÖÆÂÖÀª
			if (drawContour) {
				Scalar color(0, 255, 0);  // ÂÌÉ«ÂÖÀª
				drawContours(SignImg, vector<vector<Point>>{bar.contour}, 0, color, 2);
			}

			// ÏÔÊ¾ĞÅÏ¢
			if (showInfo) {
				Point center = bar.rect.center;
				string info = "A:" + to_string(int(bar.area)) + "-R:" + to_string(bar.WHR).substr(0, 4);
				putText(SignImg, info, Point(center.x - 30, center.y - 10),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

				// »æÖÆÖĞĞÄµã
				circle(SignImg, center, 3, Scalar(0, 0, 255), -2);
			}
		}

		Mat CopyImg = SignImg.clone();
		// ¼È±ê¼ÇÍ¼Æ¬¿ÉÊÓ»¯
		putText(CopyImg, "SignImg", Point(10, 30), 3, 1, Scalar(255), 2);
		imshow("SignImg", CopyImg);
		cout << "°´ÈÎÒâ¼ü¼ÌĞø£¬»ò°´ESCÍË³ö½ø³Ì..." << endl;
		int key = cv::waitKey(0);  // µÈ´ı°´¼ü
		// ¼ì²éÊÇ·ñ°´ÁËESC¼ü
		if (key == 27) {
			exit(0);
		}
		return SignImg;
	}

	// func:µÚÒ»°æÍêÕûµÄµÆÌõ¼ì²âÁ÷³Ì£¨»ùÓÚ»Ò¶ÈÍ¼£©£¨±¾º¯Êı·¶Î§²ÎÊıÓë SelectLightBar º¯ÊıµÄ²ÎÊı½¨Òé±£³ÖÒ»ÖÂ£©
	void ProcessLightBarV1(const string& outputPath, double minArea = 20, double maxArea = 5000,
		double minWHR = 2.0, double maxWHR = 30.0) const {

		CheckImageLoaded();
		cout << "Ãæ»ı·¶Î§: " << minArea << " ~ " << maxArea << endl;
		cout << "É«¿é¿í¸ß±È·¶Î§: " << minWHR << " ~ " << maxWHR << endl;
		Mat resultImg = MarkLightBar(SelectLightBar(minArea, maxArea, minWHR, maxWHR));
		Mat CopyImg = resultImg.clone();

		// ±£´æ½á¹û
		if (outputPath.empty()) {
			throw invalid_argument("ÎÄ¼şÂ·¾¶Îª¿Õ£¡");
		}
		else{
			imwrite(outputPath, resultImg);
			cout << "½á¹ûÒÑ±£´æÖÁ: " << outputPath << endl;
		}
		
		cout << ">>> µÆÌõ¼ì²âÍê³É <<<" << endl;
	}

	// func:µÚ¶ş°æÍêÕûµÄµÆÌõ¼ì²âÁ÷³Ì£¨»ùÓÚHSVºìÉ«Çø¼ä£©
	void ProcessLightBarV2(const string& outputPath, double minArea = 20, double maxArea = 5000,
		double minWHR = 1.2, double maxWHR = 30.0, double minH = -20.0, double maxH = 10.0,
		double minS = 100, double maxS = 255, double minV = 100, double maxV = 255) {

		CheckImageLoaded();

		// HSVµÆÌõ¼ì²â
		auto hsvselection = SelectLightBarHSV(minArea, maxArea, minWHR, maxWHR, minH, maxH, minS, maxS, minV, maxV);

		// ±ê¼ÇµÆÌõ
		Mat markedImg = MarkLightBar(hsvselection);

		// ±£´æ½á¹û
		if (outputPath.empty()) {
			throw invalid_argument("ÎÄ¼şÂ·¾¶Îª¿Õ£¡");
		}
		else {
			imwrite(outputPath, markedImg);
			cout << "½á¹ûÒÑ±£´æÖÁ: " << outputPath << endl;
		}
		cout << ">>> HSVµÆÌõ¼ì²âÍê³É <<<" << endl;

	}

	// ¸´ÖÆ¹¹Ôìº¯Êı
	ImageProcesser(const ImageProcesser& other)
		: ImgPath(other.ImgPath), Img(other.Img.clone()), isLoaded(other.isLoaded) {}

	// Îö¹¹º¯Êı
	~ImageProcesser() {}
};


int main() {
	ImageProcesser A("images/hero.png");
	//A.ShowImage();
	A.ProcessLightBarV2("copy.png");
	return 0;
}
