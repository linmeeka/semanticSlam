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

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
//class Model;
using namespace ORB_SLAM2;
//namespace ORB_SLAM2
//{
class ModelManager  
{
public:
    ModelManager()= default;
    explicit ModelManager(int max_obj_num, int lost_num_thr, float match_thr)
    {
        mMaxObjNum = max_obj_num;
        mLostNumThr = lost_num_thr;
        mMatchTHr = match_thr;
    };

    int UpdateObjectInstances(KeyFrame* kf, std::vector<std::shared_ptr<SegData>>& SegDatas);
    void UpdateObjectPointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask, PointCloud::Ptr &globalModel);
    //int InsertNewObject(std::shared_ptr<Model> obj);
    //int MatchObjectInstances(std::shared_ptr<Model> obj,Frame& pCurrentFrame,const Frame& nLastFrame);
    //int BuildNewObjects(const cv::Mat& img,const Frame& pCurrentFrame,const std::vector<std::shared_ptr<SegData>>& pImgObjsInfo);

private:
    // obj 队列
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
    //void ObjectProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> & vpPoints, std::vector<cv::Point>& imPoints);
    float CalculateIOUbyRoI(const cv::Rect& a, const cv::Rect& b);
    float CalculateIOUbyMask(const cv::Rect& a, const cv::Rect& b);
    void BuildNewModel(const long unsigned int kf_index, const std::shared_ptr<SegData>& segData);
    //float CalculateDist(const cv::Rect& a, const cv::Rect& b);
};

//}

#endif