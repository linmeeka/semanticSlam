/**
* This file is added by kylin
*/

#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>
#include "SegData.h"
#include <map>
#include <set>
#include <unordered_map>
//using namespace std;
namespace ORB_SLAM2
{
class Model
{
public:
    // explicit Model(const int index);
    // explicit Model(const int index, const long unsigned int kf_index, const std::shared_ptr<ImgObjectInfo>& pImgObjectInfo);
    // explicit Model(const int index,const std::shared_ptr<ImgObjectInfo>& pImgObjectInfo, \
    // const long unsigned int kf_index,const std::vector<MapPoint*>& pvMapPoints);

    // void UpdateObjectInfo(const std::shared_ptr<ImgObjectInfo>& pImgObjectInfo, \
    // const long unsigned int kf_index,const std::vector<MapPoint*>&pvMapPoints);

    // void UpdateObjectInfo(const std::shared_ptr<ImgObjectInfo>& pImgObjectInfo, \
    // const long unsigned int kf_index);
    Model();
    void UpdateModel();

    int GetClassId()
    {
        return mClassId;
    }
    
    int GetObjIndex()
    {
        return mIndex;
    }
    // cv::Rect GetLastRect()
    // {
    //     if(mvKeyframeIndexes.size() > 0)
    //     {
    //         return mmKeyFrameObjectInfo[mvKeyframeIndexes[mvKeyframeIndexes.size()-1]]->bbox;
    //     }
    // }
    cv::Rect GetRoI()
    {

    }
    long unsigned int GetLastFrameID()
    {
        if(mvKeyframeIndexes.size()>0)
            return mvKeyframeIndexes[mvKeyframeIndexes.size()-1];
    }


private:
    int mIndex;
    int mClassId;
    std::vector<long unsigned int> mvKeyframeIndexes;
    std::set<long unsigned  int>mvMapPointIndexes;
    std::unordered_map<long unsigned int,std::shared_ptr<SegData>> mmKeyFrameObjectInfo; //KFIndex -> Objs
    //std::unordered_map<long unsigned int,std::shared_ptr<ImgObjectInfo>> mmKeyFrameObjectInfo;
    
};

}

#endif