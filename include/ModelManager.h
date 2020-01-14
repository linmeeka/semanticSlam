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
    explicit ModelManager(int max_obj_num, int lost_num_thr)
    {
        mMaxObjNum = max_obj_num;
        //mLostNumThr = lost_num_thr;
    };

    int UpdateObjectInstances(const KeyFrame* kf, const std::vector<std::shared_ptr<SegData>> &SegDatas,const cv::Mat &Mask);
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
;
    float mMatchTHr_moving = 0.99;
    float mMatchTHr_static = 0.5;
    int mLostNumThr_moving = 1;
    int mLostNumThr_static = 5;
    //void ObjectProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> & vpPoints, std::vector<cv::Point>& imPoints);
    float CalculateIOUbyMask(const cv::Rect& roi1, const cv::Rect& roi2, const cv::Mat &mask1, const cv::Mat &mask2,const int &classId);
    float CalculateIOUbyRoI(const cv::Rect& a, const cv::Rect& b);
    void BuildNewModel(const long unsigned int kf_index, const std::shared_ptr<SegData>& segData,const cv::Mat &Mask);
    //float CalculateDist(const cv::Rect& a, const cv::Rect& b);
};

//}

#endif