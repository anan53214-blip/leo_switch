"""
graph模块单元测试

测试异质图构建功能
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph import HeteroGraphBuilder, FeatureExtractor, NodeFeatures, EdgeFeatures


class TestFeatureExtractor:
    """测试特征提取器"""
    
    def test_init(self):
        """测试初始化"""
        extractor = FeatureExtractor()
        assert extractor.normalize == True
        assert extractor.include_velocity == True
        
    def test_feature_dimensions(self):
        """测试特征维度获取"""
        extractor = FeatureExtractor()
        dims = extractor.get_feature_dimensions()
        
        # 检查所有维度
        assert dims['satellite_node'] == 10
        assert dims['user_node'] == 13
        assert dims['user_satellite_edge'] == 6
        assert dims['inter_satellite_edge'] == 3
        
    def test_feature_dimensions_no_velocity(self):
        """测试不包含速度时的维度"""
        extractor = FeatureExtractor(include_velocity=False)
        dims = extractor.get_feature_dimensions()
        
        assert dims['satellite_node'] == 7  # 减少3维速度
        
    def test_feature_names(self):
        """测试特征名称获取"""
        extractor = FeatureExtractor()
        names = extractor.get_feature_names()
        
        # 检查卫星特征名
        assert 'pos_x' in names['satellite_node']
        assert 'cpu_util' in names['satellite_node']
        
        # 检查用户特征名
        assert 'task_data' in names['user_node']
        assert 'handover_count' in names['user_node']
        
        # 检查边特征名
        assert 'snr' in names['user_satellite_edge']
        assert 'rvt' in names['user_satellite_edge']


class TestNodeFeatures:
    """测试节点特征数据类"""
    
    def test_creation(self):
        """测试创建节点特征"""
        sat_feat = np.random.randn(66, 10).astype(np.float32)
        user_feat = np.random.randn(5, 13).astype(np.float32)
        
        node_features = NodeFeatures(
            satellite_features=sat_feat,
            user_features=user_feat,
            satellite_feature_dim=10,
            user_feature_dim=13
        )
        
        assert node_features.satellite_features.shape == (66, 10)
        assert node_features.user_features.shape == (5, 13)


class TestEdgeFeatures:
    """测试边特征数据类"""
    
    def test_creation(self):
        """测试创建边特征"""
        us_edges = [(0, 5), (0, 10), (1, 5)]
        us_features = np.random.randn(3, 6).astype(np.float32)
        
        isl_edges = [(0, 1), (1, 2)]
        isl_features = np.random.randn(2, 3).astype(np.float32)
        
        edge_features = EdgeFeatures(
            user_satellite_edges=us_edges,
            user_satellite_features=us_features,
            inter_satellite_edges=isl_edges,
            inter_satellite_features=isl_features
        )
        
        assert len(edge_features.user_satellite_edges) == 3
        assert len(edge_features.inter_satellite_edges) == 2


class TestHeteroGraphBuilder:
    """测试异质图构建器"""
    
    def test_init(self):
        """测试初始化"""
        builder = HeteroGraphBuilder()
        assert builder.add_reverse_edges == True
        assert builder.add_self_loops == False
        
    def test_init_custom(self):
        """测试自定义初始化"""
        builder = HeteroGraphBuilder(
            add_reverse_edges=False,
            add_self_loops=True
        )
        assert builder.add_reverse_edges == False
        assert builder.add_self_loops == True
        
    def test_metapaths(self):
        """测试元路径获取"""
        builder = HeteroGraphBuilder()
        metapaths = builder.get_metapaths()
        
        # 应该有3条元路径
        assert len(metapaths) >= 3
        
        # 检查元路径格式
        for mp in metapaths:
            assert isinstance(mp, list)
            for edge in mp:
                assert len(edge) == 3  # (src_type, edge_type, dst_type)


class TestHeteroGraphData:
    """测试异质图数据结构"""
    
    def test_creation(self):
        """测试创建图数据"""
        from graph.builder import HeteroGraphData
        
        graph = HeteroGraphData()
        
        # 添加节点
        graph.node_features['satellite'] = np.zeros((66, 10))
        graph.node_features['user'] = np.zeros((5, 13))
        graph.num_nodes['satellite'] = 66
        graph.num_nodes['user'] = 5
        
        # 添加边
        edge_type = ('user', 'connect', 'satellite')
        graph.edge_index[edge_type] = (
            np.array([0, 1, 2]),
            np.array([5, 10, 15])
        )
        
        assert graph.get_node_types() == ['satellite', 'user']
        assert len(graph.get_edge_types()) == 1
        
    def test_num_edges(self):
        """测试边数量统计"""
        from graph.builder import HeteroGraphData
        
        graph = HeteroGraphData()
        graph.num_nodes['satellite'] = 66
        graph.num_nodes['user'] = 5
        
        # 添加多种边
        graph.edge_index[('user', 'connect', 'satellite')] = (
            np.array([0, 1, 2]),
            np.array([5, 10, 15])
        )
        graph.edge_index[('satellite', 'isl', 'satellite')] = (
            np.array([0, 1]),
            np.array([1, 2])
        )
        
        # 检查总边数
        assert graph.num_edges() == 5
        
        # 检查特定类型边数
        assert graph.num_edges(('user', 'connect', 'satellite')) == 3
        
    def test_to_dict(self):
        """测试转换为字典"""
        from graph.builder import HeteroGraphData
        
        graph = HeteroGraphData()
        graph.node_features['satellite'] = np.zeros((10, 5), dtype=np.float32)
        graph.num_nodes['satellite'] = 10
        graph.metadata['test'] = 'value'
        
        result = graph.to_dict()
        
        assert 'node_features' in result
        assert 'edge_index' in result
        assert 'metadata' in result
        assert result['metadata']['test'] == 'value'


class TestIntegrationWithEnvironment:
    """与环境的集成测试"""
    
    @pytest.fixture
    def env(self):
        """创建测试环境"""
        try:
            from environment import LEOSatelliteEnv, EnvConfig
            
            config = EnvConfig(
                num_users=3,
                episode_duration=60.0,
                time_step=5.0
            )
            env = LEOSatelliteEnv(config)
            env.reset()
            return env
        except ImportError:
            pytest.skip("环境模块不可用")
        except Exception as e:
            pytest.skip(f"环境创建失败: {e}")
    
    def test_build_graph(self, env):
        """测试从环境构建图"""
        builder = HeteroGraphBuilder()
        graph = builder.build(env)
        
        # 检查节点
        assert 'satellite' in graph.node_features
        assert 'user' in graph.node_features
        assert graph.node_features['satellite'].shape[0] == 66
        assert graph.node_features['user'].shape[0] == 3
        
        # 检查元数据
        assert 'num_satellites' in graph.metadata
        assert graph.metadata['num_satellites'] == 66
        
    def test_extract_features(self, env):
        """测试特征提取"""
        extractor = FeatureExtractor()
        node_features = extractor.extract_node_features(env)
        
        # 检查卫星特征维度
        assert node_features.satellite_features.shape == (66, 10)
        
        # 检查用户特征维度
        assert node_features.user_features.shape == (3, 13)
        
        # 检查值范围（归一化后应该在合理范围）
        assert np.all(np.abs(node_features.satellite_features) < 10)
        assert np.all(np.abs(node_features.user_features) < 10)
        
    def test_extract_edge_features(self, env):
        """测试边特征提取"""
        extractor = FeatureExtractor()
        edge_features = extractor.extract_edge_features(env)
        
        # 检查用户-卫星边
        assert isinstance(edge_features.user_satellite_edges, list)
        
        # 如果有可见卫星，应该有边
        if edge_features.user_satellite_edges:
            assert edge_features.user_satellite_features is not None
            assert edge_features.user_satellite_features.shape[1] == 6
            
        # 检查星间链路
        assert len(edge_features.inter_satellite_edges) > 0
        assert edge_features.inter_satellite_features.shape[1] == 3
        
    def test_graph_summary(self, env, capsys):
        """测试图摘要打印"""
        builder = HeteroGraphBuilder()
        graph = builder.build(env)
        
        builder.print_graph_summary(graph)
        
        captured = capsys.readouterr()
        assert '异质图摘要' in captured.out
        assert 'satellite' in captured.out
        assert 'user' in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
