#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <limits>
#include <algorithm>

// 使用命名空间简化代码
using json = nlohmann::json;
using Eigen::MatrixXd;
using Eigen::Vector2d;

// 计算两个点之间的距离
double distance(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return (a - b).norm();
}

// 计算二维向量的叉积
double cross(const Vector2d& a, const Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

// 计算直线与线段的交点
Vector2d line_segment_intersection(const Vector2d& a1, const Vector2d& a2, const Vector2d& b1, const Vector2d& b2, bool& intersects) {
    Vector2d r = a2 - a1; //代表线段a1a2的方向
    Vector2d s = b2 - b1; //代表线段b1b2的方向
    double rxs = cross(r, s);
    
    if (rxs == 0) {
        // 线段平行或共线
        intersects = false;
        return Vector2d(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());
    }

    double t = cross(b1 - a1, s) / rxs;
    double u = cross(b1 - a1, r) / rxs;

    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        intersects = true;
        return a1 + t * r; //如果存在交点，就返回交点
    } else {
        intersects = false;
        return Vector2d(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());
    }
}

// 分别求上下凸包并合成
std::vector<Eigen::Vector2d> compute_convex_hull(std::vector<Eigen::Vector2d> points) {
    std::sort(points.begin(), points.end(), [](const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) {
        return (p1.x() < p2.x()) || (p1.x() == p2.x() && p1.y() < p2.y());
    });

    std::vector<Eigen::Vector2d> hull;
    for (const auto& point : points) {
        while (hull.size() >= 2 && cross(hull[hull.size() - 1] - hull[hull.size() - 2], point - hull[hull.size() - 1]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(point);
    }

    size_t lower_hull_size = hull.size();
    for (auto it = points.rbegin(); it != points.rend(); ++it) {
        while (hull.size() > lower_hull_size && cross(hull[hull.size() - 1] - hull[hull.size() - 2], *it - hull[hull.size() - 1]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(*it);
    }

    hull.pop_back(); // 删除最后一个点，因为它与第一个点相同
    return hull;
}

// 计算长径（最大距离）
std::pair<double, std::pair<Vector2d, Vector2d>> cal_major_axis(const std::vector<Eigen::Vector2d>& hull) {
    double max_dist = 0;
    Vector2d p1, p2;
    for (size_t i = 0; i < hull.size(); ++i) {
        for (size_t j = i + 1; j < hull.size(); ++j) {
            double dis = distance(hull[i], hull[j]);
            if (dis > max_dist) {
                max_dist = dis;
                p1 = hull[i];
                p2 = hull[j];
            }
        }
    }
    return {max_dist, {p1, p2}};
}

//三种可能导致短径计算时没有达到目标位置
/*
    边的顺序是乱的
*/

// 计算短径
double cal_minor_axis(const std::vector<Eigen::Vector2d>& hull, const Vector2d& p1, const Vector2d& p2, Vector2d& p3, Vector2d& p4) {
    Vector2d major_axis = p2 - p1;
    Vector2d normal(-major_axis.y(), major_axis.x()); //创建法向量normal
    normal.normalize(); // 归一化

    double max_length = 0;
    for (const auto& point : hull) {
        if (cross(major_axis, point - p1) > 0) {//叉积为正表示point-p1在major_axis左侧，用于筛选主轴一侧的点
            double max_distance = 0;
            for (size_t i = 0; i < hull.size(); ++i) {//遍历凸包所有边
                size_t next_i = (i + 1) % hull.size();
                if (cross(major_axis, hull[next_i] - p1) <= 0) {//判断目标边是否在右侧
                    bool intersects;
                    Vector2d intersection = line_segment_intersection(point - 1000 * normal, point + 1000 * normal, hull[i], hull[next_i], intersects);//计算交点，延长法向量长度确保相交
                    if (intersects) {
                        double dist = distance(point, intersection);
                        if (dist > max_length) {
                            max_length = dist;
                            p3 = point;
                            p4 = intersection;
                        }
                    }
                }
            }
            
        }
    }
    return max_length;
}

int main() {
    // JSON 文件路径
    std::string file_path = "C:/code/extract_visual/src/predict_ct_chest_vr-0722.json"; // 确保路径正确

    // 输出文件路径
    std::string output_file_path = "C:/code/extract_visual/src/diameters_output.txt";
    std::ofstream output_file(output_file_path);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file: " << output_file_path << std::endl;
        return 1;
    }

    try {
        // 从 JSON 文件中提取凸包
        json data;
        std::ifstream file(file_path);
       
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << file_path << std::endl;
            return 1;
        }

        file >> data;

        // 提取 ct_nodule 数据
        if (!data.contains("ct_nodule")) {
            std::cerr << "JSON does not contain 'ct_nodule' key!" << std::endl;
            return 1;
        }

        auto ct_nodules = data["ct_nodule"];
        for (const auto& nodule : ct_nodules) {
            for (const auto& contour : nodule["contour3D"]) {
                std::vector<Eigen::Vector2d> points;
                for (const auto& point : contour["data"][0]) {
                    points.emplace_back(point[0], point[1]);
                }
                std::vector<Eigen::Vector2d> hull = compute_convex_hull(points);

                // 计算长径
                auto [major_axis_length, major_axis_points] = cal_major_axis(hull);
                Vector2d p1 = major_axis_points.first;
                Vector2d p2 = major_axis_points.second;

                // 计算短径
                Vector2d p3, p4;
                double minor_axis_length = cal_minor_axis(hull, p1, p2, p3, p4);

               

                // 输出结果到文件
                output_file << contour["sliceId"] << " "
                            << p1.x() << " " << p1.y() << " " << p2.x() << " " << p2.y() << " "
                            << p3.x() << " " << p3.y() << " " << p4.x() << " " << p4.y() << "\n";

                // 输出结果
                std::cout << "Contour sliceId " << contour["sliceId"] << ":\n";
                std::cout << "  Calculated long diameter: " << major_axis_length << "\n";
                std::cout << "  Calculated short diameter: " << minor_axis_length << "\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    output_file.close();
    return 0;
}
