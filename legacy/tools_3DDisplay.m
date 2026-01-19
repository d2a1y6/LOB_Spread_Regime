%% MATLAB 3D SHAP Surface Visualization (Fixed)
% 功能：绘制真正的 3D 地形图，Z轴高度代表 SHAP 值，xy轴为特征
% 修复：添加轴标签，强制 3D 视角，曲面拟合

clc; clear; close all;

% =========================================================================
% 1. 读取数据
% =========================================================================
filename = 'shap_3d_data.csv'; % 确保此文件在当前目录
if ~isfile(filename)
    error('未找到 shap_3d_data.csv，请先运行 Python 导出脚本！');
end

opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve'; % 保留原始列名(含特殊字符)
T = readtable(filename, opts);

% 获取列名作为标签
colNames = T.Properties.VariableNames;
label_x = colNames{1}; % 自动获取第一列名
label_y = colNames{2}; % 自动获取第二列名
label_z = colNames{3}; % 自动获取第三列名

% 提取数据矩阵
x = table2array(T(:, 1));
y = table2array(T(:, 2));
z = table2array(T(:, 3));

% =========================================================================
% 2. 曲面拟合 (Scatter -> Surface)
% =========================================================================
fprintf('正在拟合平滑曲面 (Lowess Smoothing)... \n');
% Lowess 拟合能很好地过滤金融数据的噪音，展示核心趋势
% Span 参数控制平滑度：0.1-0.3 比较合适，越大越平滑
[sf, gof] = fit([x, y], z, 'lowess', 'Span', 0.15); 

% 创建高分辨率网格
grid_density = 50; 
x_lin = linspace(min(x), max(x), grid_density);
y_lin = linspace(min(y), max(y), grid_density);
[X_grid, Y_grid] = meshgrid(x_lin, y_lin);

% 计算网格上的 Z 值
Z_grid = sf(X_grid, Y_grid);

% =========================================================================
% 3. 绘制 3D 图形
% =========================================================================
fig = figure('Color', 'w', 'Name', '3D SHAP Interaction', 'Position', [100, 100, 1000, 700]);
hold on; grid on;

% A. 绘制拟合曲面
% FaceAlpha: 透明度，EdgeColor: none 去掉网格线只看面
surf(X_grid, Y_grid, Z_grid, 'EdgeColor', 'none', 'FaceAlpha', 0.9);

% B. 绘制原始散点 (可选，为了对比)
% 稍微抬高一点点防止与面重叠闪烁
scatter3(x, y, z + 0.0001, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.1);

% =========================================================================
% 4. 视觉美化与标签
% =========================================================================

% 颜色映射 (Jet 或 Parula)
colormap(jet); 
c = colorbar;
c.Label.String = 'SHAP Value (Z-Axis)';
c.Label.FontSize = 11;

% 添加零平面 (参考基准面)
% 这是一个半透明的灰色平面，代表 SHAP=0
z_plane = zeros(size(X_grid));
surf(X_grid, Y_grid, z_plane, 'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% 设置标签 (Interpreter none 防止下划线转义)
xlabel(label_x, 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
ylabel(label_y, 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
zlabel('SHAP Interaction Effect', 'FontSize', 12, 'FontWeight', 'bold');

title(['Non-linear Interaction Surface: ' label_x ' vs ' label_y], ...
      'FontSize', 14, 'Interpreter', 'none');

% 强制 3D 视角
view(-45, 30); % Azimuth -45, Elevation 30
axis tight;

% 灯光效果 (让曲面更有立体感)
camlight left; 
lighting gouraud; % 高洛德着色，使曲面光滑

hold off;

% =========================================================================
% 5. 交互提示
% =========================================================================
fprintf('绘图完成！\n');
fprintf('操作提示：\n');
fprintf('1. 点击工具栏的 "Rotate 3D" 图标可以 360 度旋转观察。\n');
fprintf('2. 黑色半透明平面为 SHAP=0 的基准面。\n');
fprintf('3. 曲面隆起代表正向贡献，凹陷代表负向贡献。\n');