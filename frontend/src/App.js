import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Layout,
  Card,
  Row,
  Col,
  Statistic,
  Table,
  DatePicker,
  Select,
  Switch,
  Upload,
  Button,
  Spin,
  Alert,
  Radio,
  Typography,
  Space,
  Tag,
  Slider,
  message
} from 'antd';
import {
  UploadOutlined,
  LineChartOutlined,
  TableOutlined,
  ReloadOutlined,
  FilterOutlined
} from '@ant-design/icons';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import dayjs from 'dayjs';
import './App.css';

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

const API_BASE = 'http://localhost:8000';

const App = () => {
  // State management
  const [loading, setLoading] = useState(false);
  const [cacheKey, setCacheKey] = useState(null);
  const [fileYears, setFileYears] = useState([]);
  const [totalRecords, setTotalRecords] = useState(0);
  
  // Filter state
  const [filterOptions, setFilterOptions] = useState({
    subtypes: [],
    trains: [],
    years: [],
    months: []
  });
  
  const [filters, setFilters] = useState({
    start_date: dayjs().subtract(30, 'day').format('YYYY-MM-DD'),
    end_date: dayjs().format('YYYY-MM-DD'),
    subtype: 'All Categories',
    train: 'All Trains',
    year: 'All Years',
    remove_duplicates: false
  });
  
  // Data state
  const [stats, setStats] = useState(null);
  const [tableData, setTableData] = useState([]);
  const [timelineData, setTimelineData] = useState([]);
  const [selectedRow, setSelectedRow] = useState(null);
  const [duplicateInfo, setDuplicateInfo] = useState('');
  
  // UI state
  const [datePreset, setDatePreset] = useState('all');
  const [maxRows, setMaxRows] = useState(20);
  const [collapsed, setCollapsed] = useState(false);
  const [timeUnit, setTimeUnit] = useState('Daily');
  const [multiYear, setMultiYear] = useState(false);

  // API calls
  const uploadFiles = async (files) => {
    setLoading(true);
    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setCacheKey(response.data.cache_key);
      setFileYears(response.data.file_years);
      setTotalRecords(response.data.total_records);
      
      // Set initial date range from data
      if (response.data.date_range.min && response.data.date_range.max) {
        setFilters(prev => ({
          ...prev,
          start_date: response.data.date_range.min,
          end_date: response.data.date_range.max
        }));
      }
      
      message.success(`Uploaded ${files.length} file(s) successfully`);
      await loadFilterOptions(response.data.cache_key);
      
    } catch (error) {
      message.error(`Upload failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const loadFilterOptions = async (key) => {
    try {
      const response = await axios.get(`${API_BASE}/filter-options/${key}`);
      setFilterOptions(response.data);
    } catch (error) {
      message.error('Failed to load filter options');
    }
  };
  
  const loadAnalytics = useCallback(async () => {
    if (!cacheKey) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/analytics/${cacheKey}`, filters);
      setStats(response.data.stats);
      setDuplicateInfo(response.data.stats.duplicate_info);
    } catch (error) {
      message.error('Failed to load analytics');
    } finally {
      setLoading(false);
    }
  }, [cacheKey, filters]);
  
  const loadTableData = useCallback(async () => {
    if (!cacheKey) return;
    
    try {
      const response = await axios.post(
        `${API_BASE}/table/${cacheKey}?max_rows=${maxRows}`,
        filters
      );
      setTableData(response.data.rows);
    } catch (error) {
      message.error('Failed to load table data');
    }
  }, [cacheKey, filters, maxRows]);
  
  const loadTimelineData = useCallback(async (station = null, train = null) => {
    if (!cacheKey) return;
    
    try {
      const params = new URLSearchParams();
      if (station) params.append('selected_station', station);
      if (train) params.append('selected_train', train);
      
      const response = await axios.post(
        `${API_BASE}/timeline/${cacheKey}?${params}`,
        filters
      );
      
      setTimelineData(response.data.timeline);
      setMultiYear(response.data.multi_year);
      setTimeUnit(response.data.time_unit);
    } catch (error) {
      message.error('Failed to load timeline data');
    }
  }, [cacheKey, filters]);

  // Effects
  useEffect(() => {
    if (cacheKey) {
      loadAnalytics();
      loadTableData();
      loadTimelineData(selectedRow?.station, selectedRow?.train);
    }
  }, [cacheKey, filters, loadAnalytics, loadTableData, loadTimelineData, selectedRow]);

  // Event handlers
  const handleFileUpload = (info) => {
    const { fileList } = info;
    if (fileList.length > 0 && fileList.every(file => file.status === 'done' || !file.status)) {
      const files = fileList.map(file => file.originFileObj || file);
      uploadFiles(files);
    }
  };

  const handleDatePresetChange = (value) => {
    setDatePreset(value);
    const today = dayjs();
    
    switch (value) {
      case 'week':
        setFilters(prev => ({
          ...prev,
          start_date: today.subtract(7, 'day').format('YYYY-MM-DD'),
          end_date: today.format('YYYY-MM-DD')
        }));
        break;
      case 'month':
        setFilters(prev => ({
          ...prev,
          start_date: today.subtract(30, 'day').format('YYYY-MM-DD'),
          end_date: today.format('YYYY-MM-DD')
        }));
        break;
      case 'quarter':
        setFilters(prev => ({
          ...prev,
          start_date: today.subtract(90, 'day').format('YYYY-MM-DD'),
          end_date: today.format('YYYY-MM-DD')
        }));
        break;
      case 'all':
      default:
        // Keep current dates
        break;
    }
  };

  const handleRowClick = (record) => {
    if (selectedRow && selectedRow.station === record.station && selectedRow.train === record.train) {
      // Deselect if clicking same row
      setSelectedRow(null);
    } else {
      setSelectedRow({ station: record.station, train: record.train });
    }
  };

  const resetFilters = () => {
    setFilters({
      start_date: dayjs().subtract(30, 'day').format('YYYY-MM-DD'),
      end_date: dayjs().format('YYYY-MM-DD'),
      subtype: 'All Categories',
      train: 'All Trains',
      year: 'All Years',
      remove_duplicates: false
    });
    setSelectedRow(null);
    setDatePreset('month');
  };

  // Table columns
  const getTableColumns = () => {
    const baseColumns = [
      {
        title: 'Rank',
        dataIndex: 'rank',
        key: 'rank',
        width: 60,
        align: 'center'
      },
      {
        title: 'Station',
        dataIndex: 'station',
        key: 'station',
        ellipsis: true,
        sorter: (a, b) => a.station.localeCompare(b.station)
      },
      {
        title: 'Train',
        dataIndex: 'train',
        key: 'train',
        ellipsis: true,
        sorter: (a, b) => a.train.localeCompare(b.train)
      },
      {
        title: 'Complaints',
        dataIndex: 'complaints',
        key: 'complaints',
        align: 'center',
        sorter: (a, b) => a.complaints - b.complaints,
        render: (value) => value?.toLocaleString()
      }
    ];

    // Add year columns for multi-year comparison
    if (fileYears.length > 1 && filters.year === 'All Years') {
      fileYears.forEach(year => {
        baseColumns.push({
          title: year.toString(),
          key: `year_${year}`,
          align: 'center',
          render: (record) => record.year_data?.[year] || 0,
          sorter: (a, b) => (a.year_data?.[year] || 0) - (b.year_data?.[year] || 0)
        });
      });
    }

    return baseColumns;
  };

  // Chart components
  const renderTimelineChart = () => {
    if (!timelineData.length) {
      return <div style={{ textAlign: 'center', padding: '50px' }}>No timeline data available</div>;
    }

    if (multiYear) {
      // Multi-year comparison chart - keep overlapping but fix X-axis labels
      const yearGroups = timelineData.reduce((acc, point) => {
        if (!acc[point.year]) acc[point.year] = [];
        acc[point.year].push(point);
        return acc;
      }, {});

      // Get unique periods (without year) for X-axis
      const uniquePeriods = [...new Set(timelineData.map(point => point.period))];

      const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
      
      return (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              type="category"
              dataKey="period"
              domain={uniquePeriods}
            />
            <YAxis />
            <Tooltip 
              labelFormatter={(label) => `Period: ${label}`}
              formatter={(value, name) => [value?.toLocaleString(), `${name} Incidents`]}
            />
            {fileYears.map((year, index) => (
              <Line
                key={year}
                type="monotone"
                dataKey="incidents"
                data={yearGroups[year] || []}
                stroke={colors[index % colors.length]}
                strokeWidth={2}
                dot={{ r: 4 }}
                name={year}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      );
    } else {
      // Single year area chart
      return (
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={timelineData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date" 
              tickFormatter={(date) => dayjs(date).format('MMM DD')}
            />
            <YAxis />
            <Tooltip 
              labelFormatter={(date) => dayjs(date).format('MMM DD, YYYY')}
              formatter={(value) => [value?.toLocaleString(), 'Incidents']}
            />
            <Area 
              type="monotone" 
              dataKey="incidents" 
              stroke="#1f77b4" 
              fill="#1f77b4" 
              fillOpacity={0.3}
            />
          </AreaChart>
        </ResponsiveContainer>
      );
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 20px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
              🚂 Rail Complaints Analytics
            </Title>
          </Col>
          <Col>
            <Space>
              <Button icon={<ReloadOutlined />} onClick={resetFilters}>
                Reset Filters
              </Button>
              {cacheKey && (
                <Tag color="green">
                  {totalRecords?.toLocaleString()} records loaded
                </Tag>
              )}
            </Space>
          </Col>
        </Row>
      </Header>

      <Layout>
        <Sider 
          width={350} 
          collapsed={collapsed} 
          collapsible
          onCollapse={setCollapsed}
          style={{ background: '#fff', borderRight: '1px solid #f0f0f0' }}
        >
          <div style={{ padding: collapsed ? '16px 8px' : '16px' }}>
            {!collapsed && (
              <>
                {/* File Upload */}
                <Card size="small" style={{ marginBottom: 16 }}>
                  <Title level={5}>📁 Data Upload</Title>
                  <Upload
                    multiple
                    beforeUpload={() => false}
                    onChange={handleFileUpload}
                    accept=".csv"
                    maxCount={3}
                  >
                    <Button icon={<UploadOutlined />} block loading={loading}>
                      Upload CSV Files
                    </Button>
                  </Upload>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    Upload up to 3 CSV files for comparison
                  </Text>
                </Card>

                {/* Data Options */}
                <Card size="small" style={{ marginBottom: 16 }}>
                  <Title level={5}>🔧 Options</Title>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text>Remove Duplicates</Text>
                      <Switch
                        checked={filters.remove_duplicates}
                        onChange={(checked) => setFilters(prev => ({ ...prev, remove_duplicates: checked }))}
                        style={{ float: 'right' }}
                      />
                    </div>
                    {duplicateInfo && (
                      <Alert message={duplicateInfo} type="info" showIcon size="small" />
                    )}
                  </Space>
                </Card>

                {cacheKey && (
                  <>
                    {/* Filters */}
                    <Card size="small" style={{ marginBottom: 16 }}>
                      <Title level={5}><FilterOutlined /> Filters</Title>
                      
                      {/* Date Range */}
                      <div style={{ marginBottom: 12 }}>
                        <Text strong>Date Range</Text>
                        <Radio.Group
                          size="small"
                          value={datePreset}
                          onChange={(e) => handleDatePresetChange(e.target.value)}
                          style={{ width: '100%', marginTop: 4 }}
                        >
                          <Radio.Button value="all">All</Radio.Button>
                          <Radio.Button value="week">Week</Radio.Button>
                          <Radio.Button value="month">Month</Radio.Button>
                          <Radio.Button value="quarter">Quarter</Radio.Button>
                        </Radio.Group>
                        
                        <RangePicker
                          size="small"
                          value={[dayjs(filters.start_date), dayjs(filters.end_date)]}
                          onChange={(dates) => {
                            if (dates) {
                              setFilters(prev => ({
                                ...prev,
                                start_date: dates[0].format('YYYY-MM-DD'),
                                end_date: dates[1].format('YYYY-MM-DD')
                              }));
                              setDatePreset('custom');
                            }
                          }}
                          style={{ width: '100%', marginTop: 8 }}
                        />
                      </div>

                      {/* Category Filter */}
                      <div style={{ marginBottom: 12 }}>
                        <Text strong>Complaint Type</Text>
                        <Select
                          size="small"
                          value={filters.subtype}
                          onChange={(value) => setFilters(prev => ({ ...prev, subtype: value }))}
                          style={{ width: '100%', marginTop: 4 }}
                        >
                          {filterOptions.subtypes.map(type => (
                            <Option key={type} value={type}>{type}</Option>
                          ))}
                        </Select>
                      </div>

                      {/* Train Filter */}
                      <div style={{ marginBottom: 12 }}>
                        <Text strong>Train Number</Text>
                        <Select
                          size="small"
                          value={filters.train}
                          onChange={(value) => setFilters(prev => ({ ...prev, train: value }))}
                          style={{ width: '100%', marginTop: 4 }}
                        >
                          {filterOptions.trains.map(train => (
                            <Option key={train} value={train}>{train}</Option>
                          ))}
                        </Select>
                      </div>

                      {/* Year Filter */}
                      {fileYears.length > 1 && (
                        <div style={{ marginBottom: 12 }}>
                          <Text strong>Year</Text>
                          <Select
                            size="small"
                            value={filters.year}
                            onChange={(value) => setFilters(prev => ({ ...prev, year: value }))}
                            style={{ width: '100%', marginTop: 4 }}
                          >
                            {filterOptions.years.map(year => (
                              <Option key={year} value={year}>{year}</Option>
                            ))}
                          </Select>
                        </div>
                      )}
                    </Card>

                    {/* Display Options */}
                    <Card size="small">
                      <Title level={5}>⚙️ Display</Title>
                      <div style={{ marginBottom: 12 }}>
                        <Text>Max Table Rows</Text>
                        <Slider
                          min={10}
                          max={100}
                          step={10}
                          value={maxRows}
                          onChange={setMaxRows}
                          style={{ marginTop: 4 }}
                        />
                      </div>
                    </Card>
                  </>
                )}
              </>
            )}
          </div>
        </Sider>

        <Content style={{ padding: '16px', background: '#f5f5f5' }}>
          {!cacheKey ? (
            <Card style={{ textAlign: 'center', marginTop: '20vh' }}>
              <Title level={3}>Welcome to Rail Complaints Analytics</Title>
              <Text type="secondary">
                Upload at least one CSV file to begin analyzing rail complaints data.
                <br />
                You can upload up to 3 files for year-on-year comparison.
              </Text>
              <div style={{ marginTop: 20 }}>
                <Title level={4}>Required CSV columns:</Title>
                <ul style={{ textAlign: 'left', display: 'inline-block' }}>
                  <li><code>createdOn</code> - Timestamp of complaint</li>
                  <li><code>subTypeName</code> - Complaint category</li>
                  <li><code>trainStation</code> - Station name</li>
                  <li><code>trainNameForReport</code> - Train identifier</li>
                  <li><code>complaintRefNo</code> - Reference number</li>
                  <li><code>contactId</code> - User identifier (for deduplication)</li>
                  <li><code>pnrUtsNo</code> - Train number (optional)</li>
                </ul>
              </div>
            </Card>
          ) : (
            <Spin spinning={loading}>
              {/* Statistics Cards */}
              {stats && (
                <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                  <Col xs={24} sm={12} md={6} lg={4}>
                    <Card>
                      <Statistic 
                        title="Total Complaints" 
                        value={stats.total_complaints} 
                        formatter={(value) => value?.toLocaleString()}
                      />
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={6} lg={4}>
                    <Card>
                      <Statistic 
                        title="Stations Affected" 
                        value={stats.unique_stations}
                        formatter={(value) => value?.toLocaleString()}
                      />
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={6} lg={4}>
                    <Card>
                      <Statistic 
                        title="Trains Involved" 
                        value={stats.unique_trains}
                        formatter={(value) => value?.toLocaleString()}
                      />
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={6} lg={4}>
                    <Card>
                      <Statistic 
                        title="Avg Daily" 
                        value={stats.avg_daily} 
                        precision={1}
                      />
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={6} lg={4}>
                    <Card>
                      <Statistic 
                        title="Top Category" 
                        value={stats.top_category}
                        formatter={(value) => (
                          <Text ellipsis style={{ maxWidth: 120 }}>
                            {value}
                          </Text>
                        )}
                      />
                    </Card>
                  </Col>
                  {fileYears.length > 1 && (
                    <Col xs={24} sm={12} md={6} lg={4}>
                      <Card>
                        <Statistic 
                          title="Years Loaded" 
                          value={fileYears.join(', ')}
                          formatter={(value) => (
                            <Text style={{ fontSize: '14px' }}>{value}</Text>
                          )}
                        />
                      </Card>
                    </Col>
                  )}
                </Row>
              )}

              {/* Main Content */}
              <Row gutter={[16, 16]}>
                {/* Table */}
                <Col xs={24} lg={12}>
                  <Card 
                    title={
                      <Space>
                        <TableOutlined />
                        Complaints by Station and Train
                        {selectedRow && (
                          <Tag color="blue">
                            {selectedRow.station} - {selectedRow.train}
                          </Tag>
                        )}
                      </Space>
                    }
                    extra={
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        Click row to filter timeline
                      </Text>
                    }
                  >
                    <Table
                      columns={getTableColumns()}
                      dataSource={tableData}
                      size="small"
                      pagination={false}
                      scroll={{ y: 400 }}
                      rowKey={(record) => `${record.station}-${record.train}`}
                      onRow={(record) => ({
                        onClick: () => handleRowClick(record),
                        style: {
                          cursor: 'pointer',
                          backgroundColor: selectedRow && 
                            selectedRow.station === record.station && 
                            selectedRow.train === record.train 
                            ? '#e6f7ff' : undefined
                        }
                      })}
                    />
                  </Card>
                </Col>

                {/* Timeline Chart */}
                <Col xs={24} lg={12}>
                  <Card 
                    title={
                      <Space>
                        <LineChartOutlined />
                        {timeUnit} Incident Timeline
                        {multiYear && <Tag color="orange">Multi-Year</Tag>}
                      </Space>
                    }
                  >
                    {renderTimelineChart()}
                  </Card>
                </Col>
              </Row>
            </Spin>
          )}
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;