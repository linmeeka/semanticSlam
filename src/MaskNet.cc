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

void SegmentDynObject::SaveResult(const cv::Mat &maskRes, const std::vector<cv::Rect> &ROIRes, const std::vector<int> &ClassIdRes,  const std::vector<double> &ScoreRes, std::string dir, std::string name)
{
    std::string dirMask=dir+"/mask";
    if(dir.compare("no_save")!=0){
        DIR* _dir = opendir(dirMask.c_str());
        if (_dir) {closedir(_dir);}
        else if (ENOENT == errno)
        {
            const int check = mkdir(dirMask.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (check == -1) {
                std::string str = dirMask;
                str.replace(str.end() - 6, str.end(), "");
                mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            }
        }
        cv::imwrite(dirMask+"/"+name+".png",maskRes);
    }

    // std::string dirROI=dir+"/roi";
    // if(dir.compare("no_save")!=0){
    //     DIR* _dir = opendir(dirROI.c_str());
    //     if (_dir) {closedir(_dir);}
    //     else if (ENOENT == errno)
    //     {
    //         const int check = mkdir(dirROI.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    //         if (check == -1) {
    //             std::string str = dirROI;
    //             str.replace(str.end() - 6, str.end(), "");
    //             mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    //         }
    //     }
        
    //     std::ofstream outfile(dirROI+"/"+name+".txt",std::ios::trunc);
    //     if(!outfile.is_open())
    //     {
    //         std::cout<<"can not open file:"<<dirROI<<"/"<<name<<".txt"<<std::endl;
    //     }
    //     else
    //     {
    //         for(auto roi : ROIRes)
    //         {
    //             outfile<<roi.x<<" "<<roi.y<<" "<<roi.width<<" "<<roi.height<<std::endl;
    //         }
    //     }
    //     outfile.close();
    // }

    // std::string dirClassId=dir+"/classid";
    // if(dir.compare("no_save")!=0){
    //     DIR* _dir = opendir(dirClassId.c_str());
    //     if (_dir) {closedir(_dir);}
    //     else if (ENOENT == errno)
    //     {
    //         const int check = mkdir(dirClassId.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    //         if (check == -1) {
    //             std::string str = dirClassId;
    //             str.replace(str.end() - 6, str.end(), "");
    //             mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    //         }
    //     }
    //     std::ofstream outfile(dirClassId+"/"+name+".txt", std::ios::trunc);
    //     if(!outfile.is_open())
    //     {
    //         std::cout<<"can not open file:"<<dirClassId<<"/"<<name<<".txt"<<std::endl;
    //     }
    //     else
    //     {
    //         for(auto id : ClassIdRes)
    //         {
    //             outfile<<id<<std::endl;
    //         }
    //     }
    //     outfile.close();
    // }

    std::string dirObj=dir+"/obj";
    if(dir.compare("no_save")!=0){
        DIR* _dir = opendir(dirObj.c_str());
        if (_dir) {closedir(_dir);}
        else if (ENOENT == errno)
        {
            const int check = mkdir(dirObj.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (check == -1) {
                std::string str = dirObj;
                str.replace(str.end() - 6, str.end(), "");
                mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            }
        }
        
        std::ofstream outfile(dirObj+"/"+name+".txt",std::ios::trunc);
        if(!outfile.is_open())
        {
            std::cout<<"can not open file:"<<dirObj<<"/"<<name<<".txt"<<std::endl;
        }
        else
        {
            for(int i=0;i<ClassIdRes.size();i++)
            {
                auto roi=ROIRes[i];
                auto id=ClassIdRes[i];
                auto score=ScoreRes[i];
                outfile<<roi.x<<" "<<roi.y<<" "<<roi.width<<" "<<roi.height<<" "<<id<<" "<<score<<std::endl;
            }
        }
        outfile.close();
    }
}

void SegmentDynObject::ReadResult(std::vector<cv::Rect> &ROIRes, std::vector<int> &ClassIdRes, std::vector<double> &ScoreRes, std::string dir, std::string name)
{
    std::string nameROI=dir+"/roi/"+name+".txt";
    std::ifstream ROIfile(nameROI);
    if(ROIfile)
    {
        std::string line_info,input_result;
        std::vector<int> vectorString;
        while (getline (ROIfile, line_info)) // line中不包括每行的换行符
        {
            vectorString.clear();
            std::stringstream input(line_info);
            //依次输出到input_result中，并存入vectorString中
           // std::cout<<"roi line_info: "<<line_info<<std::endl;
            while(input>>input_result)
            {
                vectorString.push_back(std::stoi(input_result));
                //std::cout<<input_result<<" ";
            }
            //std::cout<<std::endl;
            ROIRes.push_back(cv::Rect(vectorString[0],vectorString[1],vectorString[2],vectorString[3]));
        }
    }
    else
    {
        std::cout<<"can not open file:"<<nameROI<<std::endl;
    }

    std::string nameClassId=dir+"/classid/"+name+".txt";
    std::ifstream ClassIdfile(nameClassId);
    if(ClassIdfile)
    {
        std::string line_info;
        while (getline (ClassIdfile, line_info)) // line中不包括每行的换行符
        {
            //依次输出到input_result中，并存入vectorString中
            //std::cout<<"class id line_info: "<<line_info<<std::endl;
            int id=std::stoi(line_info);
            //std::cout<<id<<std::endl;
            ClassIdRes.push_back(id);
        }
    }
    else
    {
        std::cout<<"can not open file:"<<nameClassId<<std::endl;
    }

}

void SegmentDynObject::ReadResultOneFile(std::vector<cv::Rect> &ROIRes, 
    std::vector<int> &ClassIdRes, std::vector<double> &ScoreRes, std::string dir, std::string name)
{
    std::string nameObj=dir+"/obj/"+name+".txt";
    std::ifstream Objfile(nameObj);
    if(Objfile)
    {
        std::string line_info,input_result;
        std::vector<int> vectorString;
        int id;
        double score;
        while (getline (Objfile, line_info)) // line中不包括每行的换行符
        {
            vectorString.clear();
            std::stringstream input(line_info);
            //依次输出到input_result中，并存入vectorString中
           // std::cout<<"roi line_info: "<<line_info<<std::endl;
            for(int i=0;i<4;i++)
            {
                input>>input_result;
                vectorString.push_back(std::stoi(input_result));
                //std::cout<<input_result<<" ";
            }
            input>>input_result;
            id=std::stoi(input_result);
            input>>input_result;
            score=std::stod(input_result);
            //std::cout<<std::endl;
            ROIRes.push_back(cv::Rect(vectorString[0],vectorString[1],vectorString[2],vectorString[3]));
            ClassIdRes.push_back(id);
            ScoreRes.push_back(score);
        }
    }
    else
    {
        std::cout<<"can not open file:"<<nameObj<<std::endl;
    }
}


cv::Mat SegmentDynObject::GetMaskResult(std::string dir, std::string name)
{
    PyObject* py_mask_image = PyObject_GetAttrString(this->py_module,"current_segmentation");
    cv::Mat maskRes = cvt->toMat(py_mask_image).clone();
    //cv::imwrite("seg.png",maskRes);
    maskRes.cv::Mat::convertTo(maskRes,CV_8U);//0 background y 1 foreground
    // if(dir.compare("no_save")!=0){
    //     DIR* _dir = opendir(dir.c_str());
    //     if (_dir) {closedir(_dir);}
    //     else if (ENOENT == errno)
    //     {
    //         const int check = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    //         if (check == -1) {
    //             std::string str = dir;
    //             str.replace(str.end() - 6, str.end(), "");
    //             mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    //         }
    //     }
    //     cv::imwrite(dir+"/"+name,maskRes);
    // }
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


std::vector<int> SegmentDynObject::GetClassResult(){
    std::vector<int> result;
    assert(result.size() == 0);
    PyObject* pClassList = PyObject_GetAttrString(this->py_module,"current_class_ids");
    if(!PySequence_Check(pClassList)) throw std::runtime_error("pClassList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pClassList);
    result.reserve(n);
    //result.reserve(n+1);
    //result.push_back(0); // Background
    for (int i = 0; i < n; ++i) {
        PyObject* o = PySequence_GetItem(pClassList, i);
        //assert(PyLong_Check(o));
        result.push_back(PyLong_AsLong(o));
        Py_DECREF(o);
    }
    Py_DECREF(pClassList);
    return result;
}

std::vector<double> SegmentDynObject::GetScoreesult(){
    std::vector<double> result;
    assert(result.size() == 0);
    PyObject* pClassList = PyObject_GetAttrString(this->py_module,"current_score");
    if(!PySequence_Check(pClassList)) throw std::runtime_error("pClassList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pClassList);
    result.reserve(n);
    //result.reserve(n+1);
    //result.push_back(0); // Background
    for (int i = 0; i < n; ++i) {
        PyObject* o = PySequence_GetItem(pClassList, i);
        //assert(PyLong_Check(o));
        result.push_back(PyFloat_AsDouble(o));
        Py_DECREF(o);
    }
    Py_DECREF(pClassList);
    return result;
}

void SegmentDynObject::GetSegmentation(cv::Mat &image, cv::Mat &maskRes, std::vector<cv::Rect> &ROIRes, std::vector<int> &ClassIdRes, std::vector<double> &ScoreRes, std::string dir, std::string name){
    std::string nameMask=dir+"/mask/"+name+".png";
    maskRes = cv::imread(nameMask,CV_LOAD_IMAGE_UNCHANGED);
    //maskRes = cv::imread(dir+"/"+name,CV_LOAD_IMAGE_UNCHANGED);
    if(maskRes.empty()){
        PyObject* py_image = cvt->toNDArray(image.clone());
        assert(py_image != NULL);
        PyObject_CallMethod(this->net, const_cast<char*>(this->get_seg_res.c_str()),"(O)",py_image);
        std::cout<<"detect finish"<<std::endl;
        maskRes=GetMaskResult(dir,name);
        ROIRes=GetROIResult();
        ClassIdRes=GetClassResult();
        ScoreRes=GetScoreesult();
        std::cout<<"get res finish"<<std::endl;
        SaveResult(maskRes,ROIRes,ClassIdRes,ScoreRes,dir,name);
        std::cout<<"save finish"<<std::endl;
        // for(auto rect:ROIRes)
        // {
        //     cv::rectangle(maskRes, rect, cv::Scalar(255, 0, 0),1);
        // }
    }
    else
    {
        //ReadResult(ROIRes,ClassIdRes,ScoreRes,dir,name);
        ReadResultOneFile(ROIRes,ClassIdRes,ScoreRes,dir,name);
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
