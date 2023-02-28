#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <queue>
#include <chrono>
#include <functional>
#include "WindNetPredictDetect.h"
#include <X11/Xlib.h>

using namespace cv;
using namespace std;
using namespace std::chrono; // calc fps

double fps()
{
    static double fps = 0.0;
    static int frameCount = 0;
    static auto lastTime = system_clock::now();
    static auto curTime = system_clock::now();

    curTime = system_clock::now();

    auto duration = duration_cast<microseconds>(curTime - lastTime);
    double duration_s = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    if (duration_s > 2)//2秒之后开始统计FPS
    {
        fps = frameCount / duration_s;
        frameCount = 0;
        lastTime = curTime;
    }
    ++frameCount;
    return fps;
}

void frame_write(vector<Mat>& frame_buffer) {
    cout << "this is write" << endl;
    Mat input, blob;
    VideoCapture capture;
    capture.open("rtsp://admin:a123456789@192.168.8.33:554/h264/ch1/main/av_stream/1");
    if (capture.isOpened())
    {
        cout << "Capture is opened" << endl;
        for (;;)
        {
            capture >> input;
            if (input.empty()){
				std::cout << "image is empty " << std::endl;
                continue;
				}
			if (frame_buffer.size() < 100){
				//std::cout << "frame_buffer get input" << std::endl;
                frame_buffer.push_back(input);
			}
			else {
                cout << "thread ==============> after read stop, frame_buffer.size() > 100 , write stop";
                return;
            }
        }
    }
    else {
        cout << "open camera failed" << endl;
    }
}
void frame_read(vector<Mat>& frame_buffer) {
    cout << "this is read" << endl;
    Mat frame;

    clock_t start, end, end1, end2;
    void* p_algorithm;
    p_algorithm = (void*)(new WindNetDetect());
    //std::string net_bins = "./models/hook_run106.bin";
    //std::string net_paras = "./models/hook_run106.param";
    std::string net_bins = "./models/hook_0113.bin";
    std::string net_paras = "./models/hook_0113.param";
    int init_res = ((WindNetDetect*)p_algorithm)->init1(net_bins, net_paras);
    WindNetDetect* tmp = (WindNetDetect*)(p_algorithm);
    //delete p_algorithm;
    
    
    //cv::VideoWriter writer;
	//double fps = 24.0;
	//int codec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
	//std::string fliename = "./hook.avi";
	//writer.open(fliename, codec, fps, cv::Size(2560,1440), true);
	
    //int name = 1;
    
    while (1) {
        if (!frame_buffer.empty()) {
            frame = frame_buffer.back();
            if(!frame.empty()){
				frame_buffer.pop_back();
				// 在这里加上目标检测算法
				/*
				 *
				 */
				 
				 //writer << frame;
				 
				 
				std::vector<Object> objects;
				start = clock();
				tmp->detect(frame, objects);
				//cv::imwrite("./data/" + std::to_string(name) + ".jpg", frame);
				//name++;
				//tmp->draw_objects(frame, objects);
				if (objects.size() > 0){
					int max_i = 0;
					int max_prob = 0;
					for (size_t i = 0; i < objects.size(); i++)
					{
					    const Object& obj = objects[i];
					    if (obj.prob > max_prob){
					        max_prob = obj.prob;
					        max_i = i;
					    }
					}
					
					const Object& obj_max = objects[max_i];

					fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj_max.label, obj_max.prob,
					    obj_max.rect.x, obj_max.rect.y, obj_max.rect.width, obj_max.rect.height);
					

					cv::rectangle(frame, obj_max.rect, cv::Scalar(255, 0, 0), 3);

						//detection label
					char text[256];
					//sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
					sprintf(text, "%s", class_names[obj_max.label]);

					int baseLine = 0;
					cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

					int x = obj_max.rect.x;
					int y = obj_max.rect.y - label_size.height - baseLine;
					if (y < 0)
					    y = 0;
					if (x + label_size.width > frame.cols)
					    x = frame.cols - label_size.width;

					cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
					    cv::Scalar(255, 255, 255), -1);

					cv::putText(frame, text, cv::Point(x, y + label_size.height),
					    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
				}
				
				objects.clear();
				end = clock();
				float rumtime = (float)(end - start) / CLOCKS_PER_SEC;
				std::stringstream buf;
				buf.precision(3);//覆盖默认精度
				buf.setf(std::ios::fixed);//保留小数位
				buf << rumtime;
				std::string strtime;
				strtime = buf.str();
				std::cout << "strtime22222 = " << strtime << std::endl;
			}

            if (frame_buffer.size() > 10) { // 隔帧抽取一半删除
				auto iter = frame_buffer.erase(frame_buffer.begin(), frame_buffer.end() - 5);
                //auto iter = frame_buffer.begin();
                //for (int inde = 0; inde < frame_buffer.size() / 2; inde++)
                //    frame_buffer.erase(iter++);
            }
            //cout << "FPS:" << fps() << endl;
            //imshow("Thread Sample", frame);
            //if (waitKey(10) == 113) // ’q‘ ASCII == 113
             //   break;
        }
    }
    cout << "thread ==============> read stop" << endl;
}

int main(int argc, char** argv)
{
    vector<Mat> frame_buffer;
    XInitThreads();
    std::thread tw(frame_write, ref(frame_buffer)); // pass by value
    std::thread tr(frame_read, ref(frame_buffer)); // pass by value

    tw.join();
    tr.join();

    return 0;
}
