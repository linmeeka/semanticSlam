/**
* This file is added by kylin
*/

#ifndef SEGDATA_H
#define SEGDATA_H

#include <vector>
#include <opencv2/opencv.hpp>
namespace ORB_SLAM2
{
class SegData
{
public:
    cv::Rect mImROI; // ROI
    std::vector<cv::Point2f> T_M; // 外点
    int KeyPointNum; // bouding box内关键点数目
    float weight; // 类别权重
    int classId;
    bool IsMove;
    SegData(){
        KeyPointNum=0;
    }
    void setMoveTrue()
    {
        IsMove=true;
    }
    void setMoveFalse()
    {
        IsMove=false;
    }
    // static std::vector<float> labelWeight;
    // static std::vector<cv::Scalar> labelColor;
};

/*
person:1
*/

// std::vector<float> SegData::labelWeight={
//         /*0 */    0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
//         /*10*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
//         /*20*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
//         /*30*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
//         /*40*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
//         /*50*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4,0.0,0.0,
//         /*60*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
//         /*70*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
//         /*70*/    0.0   
//             };
// std::vector<cv::Scalar> SegData::labelColor={
//         /*0*/    cv::Scalar(0,0,0),cv::Scalar(139,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
//                 cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
//         /*10*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
//                 cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
//         /*20*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
//                 cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
//         /*30*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
//                 cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
//         /*40*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
//                 cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
//         /*50*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
//                 cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
//         /*60*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,128),cv::Scalar(0,0,0),
//                 cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,128,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
           
//         /*70*/    cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
//                 cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),cv::Scalar(0,0,0),
        
//         /*80*/    cv::Scalar(0,0,0)
//             };
}

#endif