import java.io.{File, PrintWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.{Set => imutSet}
import scala.collection.mutable.ListBuffer

object UserBasedCF {
  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()
    val sparkcont = new SparkConf().setAppName("User Based Collaborative Filtering").setMaster("local")
    var sc = new SparkContext(sparkcont)
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val inputfilepath = args(0)
    val inputtestFilePath = args(1)
    val outputfilepath = args(2)
    val inputdata = sc.textFile(inputfilepath)
    var header = inputdata.first()
    val testdata = sc.textFile(inputtestFilePath)


    val trainrdd = inputdata.subtract(testdata)
    val ratingtrain = trainrdd.filter(line => line != header).map(_.split(',')).map { case (x) => (x(0).toInt, (x(1).toInt, x(2).toDouble)) }.groupByKey().sortByKey().map { case (k, v) => (k, v.toMap) }.collectAsMap()
    val UserRating = trainrdd.filter(line => line != header).map(_.split(',')).map { case (c) => (c(0).toInt, (c(2).toDouble)) }.groupByKey().sortByKey().map { case (k, v) => (k, v.toList) }.collect()
    var averageRating: Map[Int, Double] = Map()
    for (user <- UserRating) {
      var userid = user._1
      var sum = (user._2).sum
      var len = (user._2).size
      var avg = (sum / len)
      averageRating += userid -> avg
    }

    def pearsonFunction(user1: Map[Int, Double], user2: Map[Int, Double]): Double = {
      var result1 = imutSet.empty[Tuple2[Int, (Double, Double)]]
      var user_first = 0.0;
      var user_second = 0.0;
      var n = 0;
      var pc = 0.0
      for (i <- user1) {
        for (j <- user2) {
          if (i._1 == j._1) {
            user_first += i._2
            user_second += j._2
            n += 1
            result1 += Tuple2(i._1, (i._2, j._2))
          }
        }
      }
      if (!result1.isEmpty) {
        var num = 0.0;
        var den1 = 0.0;
        var den2 = 0.0;
        var den = 0.0
        val user1avgrating = user_first / n
        val user2avgrating = user_second / n
        for (input <- result1) {
          num += (input._2._1 - user1avgrating) * (input._2._2 - user2avgrating)
          den1 += math.pow((input._2._1 - user1avgrating), 2)
          den2 += math.pow((input._2._2 - user2avgrating), 2)
        }
        den = math.pow(den1, 0.5) * math.pow(den2, 0.5)
        if (den != 0) {
          pc = num / den
        }
      }
      pc
    }

    val product_user = trainrdd.filter(line => line != header).map(_.split(',')).map { case (c) => (c(1).toInt, (c(0).toInt, c(2).toDouble)) }.groupByKey().sortByKey().map { case (k, v) => (k, v.toMap) }.collectAsMap()
    var corr_user: Map[Int, Map[Int, Double]] = Map()
    var userCorrelation: Map[Int, Double] = Map()
    for (u <- ratingtrain) {
      for (u2 <- ratingtrain) {
        if (u._1 != u2._1 && u._2.size > 1 && u2._2.size > 1) {
          var u1product = (u._2.keys).toSet
          var u2product = (u2._2.keys).toSet
          var value_intersect = u1product.intersect(u2product)
          if (value_intersect.size > 1) {
            var r11 = ListBuffer.empty[Double]
            var r12 = ListBuffer.empty[Double]
            for (temp <- value_intersect) {
              r11 += u._2(temp)
              r12 += u2._2(temp)}
            if (r11.distinct.length != 1 && r12.distinct.length != 1) {
              val pearsonval = pearsonFunction(u._2, u2._2)
              userCorrelation += u2._1 -> pearsonval
              corr_user += u._1 -> userCorrelation
            }
          }
        }
      }
    }
    // println("hello")
    // println(corr_user.size)
    val rating_test = testdata.filter(!_.contains("rating")).map(_.split(',')).map { case (c) => (c(0).toInt, c(1).toInt) }.map { case (k, v) => (k, v) }.collect()
    var predict_rating: Map[(Int, Int), Double] = Map()
    for (up <- rating_test) {
      var corrated_user_set = imutSet.empty[Int]
      for (user <- corr_user) {
        if (up._1 == user._1) {
          var test_Users = imutSet.empty[Int]
          var userRating = Map.empty[Int, Double]
          var userweight = user._2
          corrated_user_set = user._2.keySet
          for (i <- product_user) {
            if (up._2 == i._1) {
              userRating = i._2
              test_Users = i._2.keySet
            }
          }
          var Userweight: Map[Int, (Double, Double)] = Map()
          for (i <- test_Users.intersect(corrated_user_set)) {
            var useWeight = userweight(i)
            var useRating = userRating(i)
            Userweight += i -> (userweight(i), userRating(i))
          }
          var predicttestrate = predictRating(Userweight, up._1, up._2, averageRating)
          predict_rating += (up._1, up._2.toInt) -> predicttestrate
        }
      }
    }
    //println(predict_rating.size)
    for(i <- rating_test){
      if(!(predict_rating.contains(i))){
        predict_rating += (i) -> averageRating(i._1)
      }
    }

    var testrate = testdata.filter(!_.contains("rating")).map(_.split(',')).map { case (c) => ((c(0).toInt, c(1).toInt), c(2).toDouble) }.map { case (k, v) => (k, v) }.collect()
    var temprate: Map[(Int, Int), (Double, Double)] = Map()
    for (userProd1 <- predict_rating){
      for(userProd2 <- testrate ){
        if(userProd1._1 == userProd2._1 ){temprate += userProd1._1 -> (userProd1._2, userProd2._2)}
      }
    }
    //temprate.foreach(println)
    var finalresult = sc.parallelize(temprate.toSeq)
    var predictROutput = sc.parallelize(predict_rating.toSeq)

    var rateDiff = finalresult.map { case ((user, product), (r1, r2)) => math.abs(r1 - r2) }
    var r1 = rateDiff.filter { case (diff) => (diff) >= 0 && diff < 1 }.count()
    var r2 = rateDiff.filter { case (diff) => (diff) >= 1 && diff < 2 }.count()
    var r3 = rateDiff.filter { case (diff) => (diff) >= 2 && diff < 3 }.count()
    var r4 = rateDiff.filter { case (diff) => (diff) >= 3 && diff < 4 }.count()
    var r5 = rateDiff.filter { case (diff) => diff >= 4 && diff <= 5 }.count()
    var r6 = rateDiff.filter { case (diff) => diff > 5 || diff < 0 }.count()
    println(">=0 and <1:" + r1)
    println(">=1 and <2:" + r2)
    println(">=2 and <3:" + r3)
    println(">=3 and <4:" + r4)
    println(">=4:" + r5)

    val MSE = finalresult.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean();
    val RMSE = math.sqrt(MSE);
    println("RMSE: " + RMSE)

    var filename1 = outputfilepath + "Aayush_Sinha_UserBasedCF.txt"
    val writefile1 = new PrintWriter(new File(filename1))
    val outputfile1 = predictROutput.sortBy { case ((user, product), predRate) => (user, product, predRate) }.map { case ((user, product), predRate) => user + "," + product + "," + predRate }
    for (a <- outputfile1.collect()) {
      writefile1.write(a + "\n")
    }
    writefile1.close()
    val end_time = System.currentTimeMillis()
    println("Time: " + (end_time - start_time) / 1000 + " secs")

  }
  def predictRating(UserRatingWeight: Map[Int, (Double, Double)], testuser: Int, testproduct: Int, averageRating:Map[Int, Double]): Double = {
    var num2 = 0.0
    var den = 0.0
    var prate = 0.0
    for (user <- UserRatingWeight) {
      num2 += ((user._2._2 - averageRating(testuser)) * user._2._1)
      den += math.abs(user._2._1)
    }
    if(den != 0){
      prate = averageRating(testuser) + (num2/den)}
    else
    {
      prate = averageRating(testuser)
    }
    prate
  }
}