/**
* This file is added by kylin
*/

#ifndef MODELMANAGER_H
#define MODELMANAGER_H

#include <vector>
#include <set>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include "Model.h"
#include "Frame.h"

class Model;

namespace ORB_SLAM2
{
class ModelManager  
{
public:
    // ObjectManager()= default;
    // explicit ObjectManager(int max_obj_num, int lost_num_thr, float match_thr)
    // {
    //     mMaxObjNum = max_obj_num;
    //     mLostNumThr = lost_num_thr;
    //     mMatchTHr = match_thr;
    // };
    ModelManager()
    {

    }

    int UpdateObjectInstances(Frame &pCurrentFrame, std::vector<std::shared_ptr<SegData>>& pImgObjsInfo);
    int InsertNewObject(std::shared_ptr<Model> obj);
    int MatchObjectInstances(std::shared_ptr<Model> obj,Frame& pCurrentFrame,const Frame& nLastFrame);
    int BuildNewObjects(const cv::Mat& img,const Frame& pCurrentFrame,const std::vector<std::shared_ptr<SegData>>& pImgObjsInfo);

private:
    std::vector<std::shared_ptr<Model>> mvObjectIntances;
    std::unordered_map<int,std::shared_ptr<Model>> mDroppedInstances;
    std::unordered_map<int,std::shared_ptr<Model>> mTrackingInstances;
    int mMaxObjNum;
    std::set<int> msObjectClasses;
    std::unordered_map<int,std::vector<int>> mClassInstanceIdMap;//class->list of instance id
    std::unordered_map<int,int> mIndexLostnumMap;
    bool mFirstFrame = true;
    int mCurrentObjectIndex = 0;
    int mLostNumThr = 5;
    float mMatchTHr = 0.3;
    void ObjectProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> & vpPoints, std::vector<cv::Point>& imPoints);
    void CaculateWorldPoints(KeyFrame* pKF, cv::Mat Scw, const std::vector<cv::Point>& imPoints, std::vector<MapPoint*> &vpPoints, cv::Mat & depth);
    float CalculateIOU(const cv::Rect& a, const cv::Rect& b);
    //float CalculateDist(const cv::Rect& a, const cv::Rect& b);
    
};

}

#endif