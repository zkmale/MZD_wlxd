#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <queue>
#include <chrono>
#include <functional>
#include "WindNetPredictDetect.h"
#include <X11/Xlib.h>
#include <json.h>

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
			if (frame_buffer.size() < 1000000000){
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
    //***************************************************************
    //***************************************************************
    //读取配置文件
    Json::Reader reader;
	Json::Value root;
    std::ifstream in("./output.json", std::ios::binary);
    if (!in.is_open())
	{
		std::cout << "Error opening file\n";
	}
    std::string net_bins;
    std::string net_paras;
    std::string label;
    int smoothing_factor = 5;
    std::string warn = "warn";
    std::string early_warn = "early_warn";
    std::string abnormal = "abnormal";

    float img_ratio = 0.5;
    float standard_b = 558.0;
    float standard_k = 0.086;
    int send_interval = 10;
    int abnormal_detect_threadshold = 150;
    int abnormal_detect_w_h = 70;

    float mild_offset = 2.2;
    float moderate_offset = 3.0;
    float serious_offset = 3.5;

    int is_save;
    int interval;
    if (reader.parse(in, root))
	{
		net_bins = root["net_bin"].asString();
		net_paras = root["net_param"].asString();
		mild_offset = root["offset_threshold"]["mild_offset"].asFloat();
		moderate_offset = root["offset_threshold"]["moderate_offset"].asFloat();
		serious_offset = root["offset_threshold"]["serious_offset"].asFloat();

		is_save = root["is_save_img"]["is_save"].asInt();
		interval = root["is_save_img"]["interval"].asInt();
        label = root["label"].asString();

        abnormal_detect_threadshold = root["abnormal_detect_threadshold"].asInt();
        abnormal_detect_w_h = root["abnormal_detect_w_h"].asInt();
        
        smoothing_factor = root["level"]["smoothing_factor"].asInt();
        warn = root["level"]["warn"].asString();
        early_warn = root["level"]["early_warn"].asString();
        abnormal = root["level"]["abnormal"].asString();

        img_ratio = root["img_ratio"].asFloat();
        standard_k = root["standard_k"].asFloat();
        standard_b = root["standard_b"].asFloat();


        send_interval = root["send_interval"].asInt();

        //const char* net_params = net_para.c_str();
        //const char* net_bins = net_bin.c_str();

		std::cout << "net_bins =  " << net_bins << std::endl;
		std::cout << "net_params =  " << net_paras << std::endl;
		std::cout << "mild_offset =  " << mild_offset << std::endl;
		std::cout << "moderate_offset =  " << moderate_offset << std::endl;
		std::cout << "serious_offset =  " << serious_offset << std::endl;
		std::cout << "interval =  " << interval << std::endl;
		std::cout << "label =  " << label << std::endl;
		std::cout << "smoothing_factor =  " << smoothing_factor << std::endl;
		std::cout << "warn =  " << warn << std::endl;
		std::cout << "early_warn =  " << early_warn << std::endl;
		std::cout << "abnormal =  " << abnormal << std::endl;
		std::cout << "img_ratio =  " << img_ratio << std::endl;
		std::cout << "standard_k =  " << standard_k << std::endl;
		std::cout << "standard_b =  " << standard_b << std::endl;
		std::cout << "send_interval =  " << send_interval << std::endl;
		std::cout << "abnormal_detect_w_h =  " << abnormal_detect_w_h << std::endl;

    }
    else
	{
        net_bins = "./models/hook_wlxd_exp1.bin";
        net_paras = "./models/hook_wlxd_exp1.param";
        //mild_offset = 2.2;
        is_save = 0;
        interval = 1500;
        label = "crooked";
		std::cout << "parse error\n" << std::endl;
	}
	in.close();
    //*****************************************************************
    //*****************************************************************

    Mat frame;
    Mat send_frame;
    clock_t start, end, end1, end2;
    void* p_algorithm;
    p_algorithm = (void*)(new WindNetDetect());
    //std::string net_bins = "./models/hook_wlxd_exp1.bin";
    //std::string net_paras = "./models/hook_wlxd_exp1.param";
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
    //正常的标准线
    float standard_x;
    //int threshold_x;
    //判断是否歪拉斜吊的阈值
    int mild_threshold_x;
    int moderate_threshold_x;
    int serious_threshold_x;

    int hook_center_point_x;
    int hook_center_point_y;

    int x_hook;
    int y_hook;
    int w_hook;
    int h_hook;
    std::string level;
    
    int img_name = 1;
    int k = 1;
    int smoothing_warn_it = 0;
    //int smoothing_warn_it = 0;
    int smoothing_early_warn_it = 0;
    int smoothing_abnormal_it = 0;

    float distance = 0.0;

    int moderate_alarm_key = 0;
    int serious_alarm_key = 0;
    
    while (1) {
        sleep(1);
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
                int max_i = -1;
                //tmp->draw_objects(frame, objects);
                if (objects.size() > 0){
                    //contral_num = 0;
                    int max_prob = 0;
                    for (size_t i = 0; i < objects.size(); i++)
                    {
                        const Object& obj = objects[i];
                        x_hook = obj.rect.x;
                        y_hook = obj.rect.y;
                        w_hook = obj.rect.width;
                        h_hook = obj.rect.height;
                        hook_center_point_x = x_hook + w_hook/2;
                        hook_center_point_y = y_hook + h_hook/2;
                        standard_x = standard_k*hook_center_point_y + standard_b;
                        distance = std::abs(hook_center_point_x - standard_x);

                        if(distance <= abnormal_detect_threadshold && w_hook + h_hook < abnormal_detect_w_h){
                            if (obj.prob > max_prob){
                                max_prob = obj.prob;
                                max_i = i;
                            }
                        }
                    }
                    if (max_i != -1){
                        const Object& obj_max = objects[max_i];

                        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj_max.label, obj_max.prob,
                            obj_max.rect.x, obj_max.rect.y, obj_max.rect.width, obj_max.rect.height);
                        
                        x_hook = obj_max.rect.x;
                        y_hook = obj_max.rect.y;
                        w_hook = obj_max.rect.width;
                        h_hook = obj_max.rect.height;

                        hook_center_point_x = x_hook + w_hook/2;
                        hook_center_point_y = y_hook + h_hook/2;

                        //standard_x = 1218 - 0.084*hook_center_point_y;
                        standard_x = standard_k*hook_center_point_y + standard_b;
                        //threshold_x = maxs(w_hook, h_hook) * 2;

                        //******************************************************
                        //判断是否歪拉斜吊的三个阈值
                        mild_threshold_x = maxs(w_hook, h_hook) * mild_offset;
                        moderate_threshold_x = maxs(w_hook, h_hook) * moderate_offset;
                        serious_threshold_x = maxs(w_hook, h_hook) * serious_offset;

                        std::vector<std::string> label_hook;
                        label_hook.push_back(std::to_string(x_hook));
                        label_hook.push_back(std::to_string(y_hook));
                        label_hook.push_back(std::to_string(w_hook));
                        label_hook.push_back(std::to_string(h_hook));
                        std::cout << "mild_threshold_x = " << mild_threshold_x << std::endl;
                        std::cout << "moderate_threshold_x = " << moderate_threshold_x << std::endl;
                        std::cout << "serious_threshold_x = " << serious_threshold_x << std::endl;

                        std::cout << "standard_x = " << standard_x << std::endl;
                        std::cout << "hook_center_point_x = " << hook_center_point_x << std::endl;
                        distance = std::abs(hook_center_point_x - standard_x);
                        std::cout << "distance = " << distance << std::endl;

                        //std::cout << "hook_center_point_y = " << hook_center_point_y << std::endl;
                        //std::cout << "zz = " << (1218.0-hook_center_point_x)/hook_center_point_y << std::endl;

                        if(distance > serious_threshold_x){
                            if(serious_alarm_key == 1){
                                smoothing_warn_it++;
                                smoothing_early_warn_it = 0;
                                smoothing_abnormal_it = 0;
                                std::cout << "smoothing_warn_it = " << smoothing_warn_it << std::endl;
                                if(smoothing_warn_it >= smoothing_factor){
                                    cv::resize(frame, send_frame, cv::Size(1280 * img_ratio, 720 * img_ratio));
                                    if(smoothing_warn_it % send_interval == smoothing_factor){
                                        //level = "warn";
                                        std::cout << "level = " << warn << std::endl;
                                        tmp->send_json(send_frame, label, warn);
                                    }
                                }
                            }
                            
                        }
                        else if(distance > moderate_threshold_x){
                            serious_alarm_key = 1;
                            if(moderate_alarm_key == 1){
                                smoothing_warn_it = 0;
                                smoothing_early_warn_it++;
                                smoothing_abnormal_it = 0;
                                std::cout << "smoothing_early_warn_it = " << smoothing_early_warn_it << std::endl;
                                if(smoothing_early_warn_it >= smoothing_factor){
                                    cv::resize(frame, send_frame, cv::Size(1280 * img_ratio, 720 * img_ratio));
                                    if(smoothing_early_warn_it % send_interval == smoothing_factor){
                                        //level = "warn";
                                        std::cout << "level = " << early_warn << std::endl;
                                        tmp->send_json(send_frame, label, early_warn);
                                    }
                                }
                            }
                            
                        }
                        else if(distance > mild_threshold_x){
                            smoothing_warn_it = 0;
                            smoothing_early_warn_it = 0;
                            smoothing_abnormal_it++;
                            std::cout << "smoothing_abnormal_it = " << smoothing_abnormal_it << std::endl;
                            moderate_alarm_key = 1;
                            serious_alarm_key = 0;
                            if(smoothing_abnormal_it >= smoothing_factor){
                                cv::resize(frame, send_frame, cv::Size(1280 * img_ratio, 720 * img_ratio));
                                if(smoothing_abnormal_it % send_interval == smoothing_factor){
                                    //level = "warn";
                                    std::cout << "level = " << abnormal << std::endl;
                                    tmp->send_json(send_frame, label, abnormal);
                                }
                            }
                        }
                        else{
                            //std::cout << "cccccc" << std::endl;
                            
                            smoothing_warn_it = 0;
                            smoothing_early_warn_it = 0;
                            smoothing_abnormal_it = 0;

                            moderate_alarm_key = 0;
                            serious_alarm_key = 0;
                            //level = "normal";
                            //tmp->send_json(frame1, label_hook, level);
                        }
                        
                        cv::rectangle(frame, obj_max.rect, cv::Scalar(255, 0, 0), 1);

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
                    else{
                        smoothing_warn_it = 0;
                        smoothing_early_warn_it = 0;
                        smoothing_abnormal_it = 0;
                        moderate_alarm_key = 0;
                        serious_alarm_key = 0;
                    }
                }
                else{
                    smoothing_warn_it = 0;
                    smoothing_early_warn_it = 0;
                    smoothing_abnormal_it = 0;
                    moderate_alarm_key = 0;
                    serious_alarm_key = 0;
                }
            }

            if(k%interval == 2 && is_save == 1){
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
