/**
* This file is part of DynaSLAM.
*
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/

#include "MaskNet.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <dirent.h>
#include <errno.h>

namespace DynaSLAM
{

#define U_SEGSt(a)\
    gettimeofday(&tvsv,0);\
    a = tvsv.tv_sec + tvsv.tv_usec/1000000.0
struct timeval tvsv;
double t1sv, t2sv,t0sv,t3sv;
void tic_initsv(){U_SEGSt(t0sv);}
void toc_finalsv(double &time){U_SEGSt(t3sv); time =  (t3sv- t0sv)/1;}
void ticsv(){U_SEGSt(t1sv);}
void tocsv(){U_SEGSt(t2sv);}
// std::cout << (t2sv - t1sv)/1 << std::endl;}

SegmentDynObject::SegmentDynObject(){
    std::cout << "Importing Mask R-CNN Settings..." << std::endl;
    ImportSettings();
    std::string x;
    setenv("PYTHONPATH", this->py_path.c_str(), 1);
    x = getenv("PYTHONPATH");
    Py_Initialize();
    this->cvt = new NDArrayConverter();
    this->py_module = PyImport_ImportModule(this->module_name.c_str());
    assert(this->py_module != NULL);
    this->py_class = PyObject_GetAttrString(this->py_module, this->class_name.c_str());
    assert(this->py_class != NULL);
    this->net = PyInstance_New(this->py_class, NULL, NULL);
    assert(this->net != NULL);
    std::cout << "Creating net instance..." << std::endl;
    cv::Mat image  = cv::Mat::zeros(480,640,CV_8UC3); //Be careful with size!!
    std::cout << "Loading net parameters..." << std::endl;
    GetSegmentation(image);
}

SegmentDynObject::~SegmentDynObject(){
    delete this->py_module;
    delete this->py_class;
    delete this->net;
    delete this->cvt;
}

// cv::Mat SegmentDynObject::  GetSegmentation(cv::Mat &image,std::string dir, std::string name){
//     // 读取或获取分割结果
//     cv::Mat seg = cv::imread(dir+"/"+name,CV_LOAD_IMAGE_UNCHANGED);
//     if(seg.empty()){
//         PyObject* py_image = cvt->toNDArray(image.clone());
//         assert(py_image != NULL);
//         PyObject* py_mask_image = PyObject_CallMethod(this->net, const_cast<char*>(this->get_dyn_seg.c_str()),"(O)",py_image);
//         seg = cvt->toMat(py_mask_image).clone();
//         cv::imwrite("seg.png",seg);
//         seg.cv::Mat::convertTo(seg,CV_8U);//0 background y 1 foreground
//         if(dir.compare("no_save")!=0){
//             DIR* _dir = opendir(dir.c_str());
//             if (_dir) {closedir(_dir);}
//             else if (ENOENT == errno)
//             {
//                 const int check = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
//                 if (check == -1) {
//                     std::string str = dir;
//                     str.replace(str.end() - 6, str.end(), "");
//                     mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
//                 }
//             }
//             cv::imwrite(dir+"/"+name,seg);
//         }
//     }
//     return seg;
// }

cv::Mat SegmentDynObject::GetMaskResult(std::string dir, std::string name)
{
    PyObject* py_mask_image = PyObject_GetAttrString(this->py_module,"current_segmentation");
    cv::Mat maskRes = cvt->toMat(py_mask_image).clone();
    cv::imwrite("seg.png",maskRes);
    maskRes.cv::Mat::convertTo(maskRes,CV_8U);//0 background y 1 foreground
    if(dir.compare("no_save")!=0){
        DIR* _dir = opendir(dir.c_str());
        if (_dir) {closedir(_dir);}
        else if (ENOENT == errno)
        {
            const int check = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (check == -1) {
                std::string str = dir;
                str.replace(str.end() - 6, str.end(), "");
                mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            }
        }
        cv::imwrite(dir+"/"+name,maskRes);
    }
    return maskRes;
}

std::vector<cv::Rect> SegmentDynObject::GetROIResult(){
    std::vector<cv::Rect> result;
    assert(result.size() == 0);
    PyObject* pRoiList = PyObject_GetAttrString(this->py_module,"current_bounding_boxes");
    if(!PySequence_Check(pRoiList)) throw std::runtime_error("pRoiList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pRoiList);
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
        PyObject* pRoi = PySequence_GetItem(pRoiList, i);
        assert(PySequence_Check(pRoi));
        Py_ssize_t ncoords = PySequence_Length(pRoi);
        assert(ncoords==4);

        PyObject* c0 = PySequence_GetItem(pRoi, 0);
        PyObject* c1 = PySequence_GetItem(pRoi, 1);
        PyObject* c2 = PySequence_GetItem(pRoi, 2);
        PyObject* c3 = PySequence_GetItem(pRoi, 3);
        //assert(PyLong_Check(c0) && PyLong_Check(c1) && PyLong_Check(c2) && PyLong_Check(c3));

        int a = PyLong_AsLong(c0);
        int b = PyLong_AsLong(c1);
        int c = PyLong_AsLong(c2);
        int d = PyLong_AsLong(c3);
        Py_DECREF(c0);
        Py_DECREF(c1);
        Py_DECREF(c2);
        Py_DECREF(c3);

        result.push_back(cv::Rect(b,a,d-b,c-a));
        Py_DECREF(pRoi);
    }
    Py_DECREF(pRoiList);
    return result;
}

void SegmentDynObject::GetSegmentation(cv::Mat &image, cv::Mat &maskRes, std::vector<cv::Rect> &ROIRes, std::string dir, std::string name){
    maskRes = cv::imread(dir+"/"+name,CV_LOAD_IMAGE_UNCHANGED);
    if(maskRes.empty()){
        PyObject* py_image = cvt->toNDArray(image.clone());
        assert(py_image != NULL);
        PyObject_CallMethod(this->net, const_cast<char*>(this->get_seg_res.c_str()),"(O)",py_image);
        maskRes=GetMaskResult(dir,name);
        ROIRes=GetROIResult();
        // for(auto rect:ROIRes)
        // {
        //     cv::rectangle(maskRes, rect, cv::Scalar(255, 0, 0),1);
        // }
    }
}

cv::Mat SegmentDynObject::GetSegmentation(cv::Mat &image,std::string dir, std::string name){
    // 读取或获取分割结果
    cv::Mat seg = cv::imread(dir+"/"+name,CV_LOAD_IMAGE_UNCHANGED);
    if(seg.empty()){
        PyObject* py_image = cvt->toNDArray(image.clone());
        assert(py_image != NULL);
        PyObject_CallMethod(this->net, const_cast<char*>(this->get_seg_res.c_str()),"(O)",py_image);
        //PyObject* py_mask_image = getPyObject("current_segmentation"); 
        PyObject* py_mask_image = PyObject_GetAttrString(this->py_module,"current_segmentation");
        seg = cvt->toMat(py_mask_image).clone();
        cv::imwrite("seg.png",seg);
        seg.cv::Mat::convertTo(seg,CV_8U);//0 background y 1 foreground
        if(dir.compare("no_save")!=0){
            DIR* _dir = opendir(dir.c_str());
            if (_dir) {closedir(_dir);}
            else if (ENOENT == errno)
            {
                const int check = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                if (check == -1) {
                    std::string str = dir;
                    str.replace(str.end() - 6, str.end(), "");
                    mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                }
            }
            cv::imwrite(dir+"/"+name,seg);
        }
    }
    return seg;
}

void SegmentDynObject::ImportSettings(){
    std::string strSettingsFile = "./Examples/RGB-D/MaskSettings.yaml";
    cv::FileStorage fs(strSettingsFile.c_str(), cv::FileStorage::READ);
    fs["py_path"] >> this->py_path;
    fs["module_name"] >> this->module_name;
    fs["class_name"] >> this->class_name;
    fs["get_dyn_seg"] >> this->get_dyn_seg;
    fs["get_seg_res"] >> this->get_seg_res;
    // std::cout << "    py_path: "<< this->py_path << std::endl;
    // std::cout << "    module_name: "<< this->module_name << std::endl;
    // std::cout << "    class_name: "<< this->class_name << std::endl;
    // std::cout << "    get_dyn_seg: "<< this->get_dyn_seg << std::endl;
}


}
