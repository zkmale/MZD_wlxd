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
    std::string net_bins = "./models/hook_wlxd_exp1.bin";
    std::string net_paras = "./models/hook_wlxd_exp1.param";
    int init_res = ((WindNetDetect*)p_algorithm)->init1(net_bins, net_paras);
    WindNetDetect* tmp = (WindNetDetect*)(p_algorithm);
    //delete p_algorithm;
    
    
    //cv::VideoWriter writer;
	//double fps = 24.0;
	//int codec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
	//std::string fliename = "./hook.avi";
	//writer.open(fliename, codec, fps, cv::Size(2560,1440), true);

    //******************************
    //检测是否歪拉斜吊参数
    float standard_x;
    int threshold_x;
    int hook_center_point_x;
    int hook_center_point_y;

    int x_hook;
    int y_hook;
    int w_hook;
    int h_hook;
    std::string level;
    
    int img_name = 1;
    int k = 1;
    
	
    
    while (1) {
        if (!frame_buffer.empty()) {
            frame = frame_buffer.back();
            frame_buffer.pop_back();
            // 在这里加上目标检测算法
            /*
             *
             */
             
             //writer << frame;
             
             
            std::vector<Object> objects;
            start = clock();

            if(!frame.empty()){
                tmp->detect(frame, objects);
                //tmp->draw_objects(frame, objects);
                if (objects.size() > 0){
                    //contral_num = 0;
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
                    
                    
                    
                    x_hook = obj_max.rect.x;
                    y_hook = obj_max.rect.y;
                    w_hook = obj_max.rect.width;
                    h_hook = obj_max.rect.height;

                    hook_center_point_x = x_hook + w_hook/2;
                    hook_center_point_y = y_hook + h_hook/2;

                    standard_x = 1218 - 0.084*hook_center_point_y;
                    threshold_x = maxs(w_hook, h_hook) * 2;
                    std::vector<std::string> label_hook;
                    label_hook.push_back(std::to_string(x_hook));
                    label_hook.push_back(std::to_string(y_hook));
                    label_hook.push_back(std::to_string(w_hook));
                    label_hook.push_back(std::to_string(h_hook));
                    std::cout << "standard_x = " << standard_x << std::endl;
                    std::cout << "threshold_x = " << threshold_x << std::endl;
                    std::cout << "hook_center_point_x = " << hook_center_point_x << std::endl;
                    std::cout << "hook_center_point_y = " << hook_center_point_y << std::endl;
                    std::cout << "zz = " << (1218.0-hook_center_point_x)/hook_center_point_y << std::endl;

                    if(std::abs(hook_center_point_x - standard_x) > threshold_x){
                        level = "hook_offset";
                        tmp->send_json(frame, label_hook, level);
                    }


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
            }

            if(k%1500 == 2){
                cv::imwrite("./data/" + std::to_string(img_name) + ".jpg", frame);
                img_name++;
            }
            k++;

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
            //cv::imshow("image", frame);
            //cv::waitKey(10);


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
