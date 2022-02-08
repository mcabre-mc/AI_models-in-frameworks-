# Setup

## Create Table Schema

### Implementations
| Name      |
| ----------- |
| Spark      |
| Keras   |
| Scikit-learn   |
| Pytorch   |
|  Matlab Deep Learning Toolbox   |
| ML.NET   |
| Mlpack  |
| Apache Mahout (MapReduce)   |

### Classification Models
| Name      |
| ----------- |
| Random Forest   |
| Multilayer Perceptron   |

### Regression Models
| Name      |
| ----------- |
| Linear Regression   |

### Clustering Models
| Name      |
| ----------- |
| k-means   |
| LDA   |

### Datasets
| FileLink      | SchemaName [primary key] | HasHeader |
| ----------- | ----------- | ----------- |
| https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data      | Abolone      | False |

### ColumnMappings
| SchemaName [foreign key] | FileColumnName | SchemaColumnName |
| ----------- | ----------- | ----------- |
| Abolone | Sex | Sex |
| Abolone | Length | Length  |
| Abolone | Diameter | Diameter |
| Abolone | Height | Height |
| Abolone | Whole weight | WholeWeight |
| Abolone | Shucked weight | ShuckedWeight |
| Abolone | Viscera weight | VisceraWeight |
| Abolone | Shell weight | ShellWeight |
| Abolone | Rings | Rings |

### Abolone
| Sex      | Length | Diameter  |  Height  | WholeWeight  | ShuckedWeight  | VisceraWeight | ShellWeight | Rings |
| -----------      | ----------- | -----------  |  -----------  | -----------  | -----------  | ----------- | ----------- | ----------- |

### LogisticRegression
| SchemaName [foreign key] | MaxIterations | RegularParameter |
| ----------- | ----------- | ----------- |
| Abolone | 1 | 2 |

### LinearRegression
| SchemaName [foreign key] | MaxIterations | RegularParameter |
| ----------- | ----------- | ----------- |
| Abolone | 1 | 2 |

### RegressionEvaluation
| SchemaName  | EvaluationName | ImplementationName | DataLoadingTime | TrainingTime | EvlauationTime |

