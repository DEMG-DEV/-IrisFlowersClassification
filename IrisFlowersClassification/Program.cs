using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;

namespace IrisFlowersClassification
{
    public class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

        static async Task Main(string[] args)
        {
            PredictionModel<IrisData, ClusterPrediction> model = Train();
            await model.WriteAsync(_modelPath);
            var prediction = model.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
            Console.Write("press any key to close...");
            Console.ReadKey();
        }

        private static PredictionModel<IrisData, ClusterPrediction> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<IrisData>(separator: ','));
            pipeline.Add(new ColumnConcatenator(
                    "Features",
                    "SepalLength",
                    "SepalWidth",
                    "PetalLength",
                    "PetalWidth"));
            pipeline.Add(new KMeansPlusPlusClusterer() { K = 3 });
            var model = pipeline.Train<IrisData, ClusterPrediction>();
            return model;

        }
    }
}
