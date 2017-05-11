/**
 * Date:  2016
 * Author: Rafael Muñoz Salinas
 * Description: demo application of DBoW3
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"

using namespace DBoW3;
using namespace std;


//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}

void saveToFile(string filename,const vector<cv::Mat> &features){

    //test it is not created
    std::ifstream ifile(filename);
    if (ifile.is_open()){cerr<<"ERROR::: Output File "<<filename<<" already exists!!!!!"<<endl;exit(0);}
    std::ofstream ofile(filename);
    if (!ofile.is_open()){cerr<<"could not open output file"<<endl;exit(0);}
    uint32_t size=features.size();
    ofile.write((char*)&size,sizeof(size));
    for(auto &f:features){
        if( !f.isContinuous()){
            cerr<<"Matrices should be continuous"<<endl;exit(0);
        }
        uint32_t aux=f.cols; ofile.write( (char*)&aux,sizeof(aux));
          aux=f.rows; ofile.write( (char*)&aux,sizeof(aux));
          aux=f.type(); ofile.write( (char*)&aux,sizeof(aux));
        ofile.write( (char*)f.ptr<uchar>(0),f.total()*f.elemSize());
    }
    cout << "Save finished" << endl;
}

bool readFeaturesFromFile(string filename, vector<cv::Mat> & features){
    //test it is not created
    std::ifstream ifile(filename);
    if (!ifile.is_open()){
        cout << filename << " does not exist."<<endl;
        return false;
    }
    uint32_t size;
    ifile.read((char*)&size,sizeof(size));
    features.resize(size);
    for(size_t i=0;i<size;i++){

        uint32_t cols,rows,type;
        ifile.read( (char*)&cols,sizeof(cols));
        ifile.read( (char*)&rows,sizeof(rows));
        ifile.read( (char*)&type,sizeof(type));
        features[i].create(rows,cols,type);
        ifile.read( (char*)features[i].ptr<uchar>(0),features[i].total()*features[i].elemSize());
    }
    return true;
}

vector<string> readImagePaths(int argc,char **argv,int start){
    vector<string> paths;
    for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
        return paths;
}

vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="", bool loadifexist = true) throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create();
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(400, 4, 3, true);
    else if(descriptor=="sift") fdetector=cv::xfeatures2d::SIFT::create(500,8);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;


    cout << "Extracting   features..." << endl;
    features.reserve(path_to_images.size());
    if(loadifexist && readFeaturesFromFile("features."+descriptor, features))
        return features;

    double p = 0;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        //cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
//        cv::resize(image, image, cv::Size(320,240), 0.5,0.5);
//        cv::imshow(string("image") + to_string(i),image);
//        cv::waitKey(0);
        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        //cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        if(!descriptors.isContinuous())
        {
            cerr << "warning: image " << path_to_images[i] << " is wrong." << endl;
        }

        features.push_back(descriptors);
        //cout<<"done detecting features"<<endl;
        if((int)i == int((int)path_to_images.size() * p))
        {
            cout << p * 100 << "% finished from " << path_to_images.size() << " images" << endl;
            p += 0.01;
        }
    }
    saveToFile("features."+descriptor, features);
    return features;
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<cv::Mat> &features, int k = 9, int L = 3, string name = "small_voc")
{
    // branching factor and depth levels
    //const int k = 9;
    //const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L2_NORM;

    DBoW3::Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save(name+".yml.gz");
    cout << "Done" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

//    //lets do something with this vocabulary
//    cout << "Matching images against themselves (0 low, 1 high): " << endl;
//    BowVector v1, v2;
//    for(size_t i = 0; i < features.size(); i++)
//    {
//        voc.transform(features[i], v1);
//        for(size_t j = 0; j < features.size(); j++)
//        {
//            voc.transform(features[j], v2);

//            double score = voc.score(v1, v2);
//            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
//        }
//    }
}

////// ----------------------------------------------------------------------------

void testDatabase(const  vector<cv::Mat > &features, string name = "small_voc")
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    Vocabulary voc(name+".yml.gz");

    Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 7);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
//    cout << "Saving database..." << endl;
//    db.save("small_db.yml.gz");
//    cout << "... done!" << endl;

//    // once saved, we can load it again
//    cout << "Retrieving database once again..." << endl;
//    Database db2("small_db.yml.gz");
//    cout << "... done! This is: " << endl << db2 << endl;
}


// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc<=2){
            cerr<<"Usage:  descriptor_name     image0 image1 ... \n\t descriptors:brisk,surf,orb ,akaze(only if using opencv 3)"<<endl;
             return -1;
        }

        string descriptor=argv[1];

        auto images=readImagePaths(argc,argv,2);
        vector< cv::Mat   >   features= loadFeatures(images,descriptor, true);
//        vector< cv::Mat   >   features2;
//        for(auto i = features.begin(); i !=features.end(); i++)
//            features2.push_back((*i).clone());
        testVocCreation(features, 10, 6, descriptor);

//        for(int i = 0; i < features.size(); i++)
//        {
//            for(int j = 0; j < features[i].rows; j++)
//            {
//                for(int k = 0; k < features[i].cols; k++)
//                {
//                    if(features[i].at<float>(j,k) != features2[i].at<float>(j,k))
//                    {
//                        cout << "diff in image " << i << " at row " << j << endl;
//                        //cout << features[i].row(j) << endl << features2[i].row(j) << endl;
//                        break;
//                    }
//                }
//            }
//        }
//        vector< cv::Mat   >   features2= loadFeatures(images,descriptor);
//        testDatabase(features2);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
