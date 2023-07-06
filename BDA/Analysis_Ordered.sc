import java.io.FileWriter
import java.io.BufferedWriter
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

val outputFile = "./output1.txt"
val writer = new BufferedWriter(new FileWriter(outputFile, true))

// Define a case class to represent a flight record, and a function to parse a line of the dataset
// ------------------------------------------------------------------------------------------
case class Flight(dofM: String, dofW: String, carrier: String, tailnum: String, flnum: Int, org_id: String, origin: String,
                    dest_id: String, dest: String, crsdeptime: Double, deptime: Double, depdelaymins: Double, crsarrtime: Double,
                    arrtime: Double, arrdelay: Double, crselapsedtime: Double, dist: Int)

def parseFlight(str: String): Flight = {
    val line = str.split(",")
    Flight(line(0), line(1), line(2), line(3), line(4).toInt, line(5), line(6), line(7), line(8), line(9).toDouble,
      line(10).toDouble, line(11).toDouble, line(12).toDouble, line(13).toDouble, line(14).toDouble, line(15).toDouble, line(16).toInt)
  }

// ------------------------------------------------------------------------------------------
// Load the data file
// ------------------------------------------------------------------------------------------
val inputRDD = sc.textFile("/home/venom/repo/scala-end/flight_data.csv")
val header = inputRDD.first()
val textRDD = inputRDD.filter(row => row != header)
val flightsRDD = textRDD.map(parseFlight).cache()
// ------------------------------------------------------------------------------------------

// Total Count of flights (rows in our data)
val totalFlights = flightsRDD.count()
writer.append("Total number of flights: " + totalFlights + "\n\n")

// How many flights are delayed
val totalFlights = flightsRDD.count()
val totalDelayedFlights = flightsRDD.filter(flight => flight.depdelaymins > 0).count()
val overallDelayPercentage = (totalDelayedFlights.toDouble / totalFlights.toDouble) * 100

writer.append("\n\nOverall delay percentage\n")
writer.append("Overall delay percentage = " + overallDelayPercentage.toString + "\n")
writer.append("\n")

// ------------------------------------------------------------------------------------------

// Percentage of delayed flights per carrier

val carrier_wise_delay_percentage = flightsRDD.map(flight => (flight.carrier, flight.depdelaymins)).filter(flight => flight._2 > 0).map(flight => (flight._1, 1)).reduceByKey(_ + _).mapValues(_ / totalDelayedFlights.toDouble * 100).sortBy(_._2 , ascending = false)

writer.append("\n\nCarrier wise percent of delays \n")
writer.append("Carrier\t\tPercentage of delay\n")
writer.append("---------------------\n")
carrier_wise_delay_percentage.take(10).foreach { case (carrier, percentage) =>
  writer.append(s"$carrier\t\t$percentage\n")
}
writer.append("\n")


// ------------------------------------------------------------------------------------------

//  Num of Planes in each day of week
val planesPerDayOfWeek = flightsRDD.groupBy(_.dofW).mapValues(_.size).sortBy(_._1 , ascending = true)

writer.append("\n\nNum of Planes in each day of week \n")
writer.append("Day of Week\t\tCount\n")
writer.append("---------------------\n")
planesPerDayOfWeek.take(7).foreach { case (dofW, count) =>
  writer.append(s"$dofW\t\t$count\n")
}
writer.append("\n")

// Average delay in each day of week in hours
val average_delay_per_week = flightsRDD.groupBy(_.dofW).mapValues(_.map(_.depdelaymins).sum / 60).sortBy(_._1 , ascending = true)

writer.append("\n\nAverage delay in each day of week in hours \n")
writer.append("Day of Week\t\tAverage Delay (in hours)\n")
writer.append("---------------------\n")
average_delay_per_week.take(7).foreach { case (dofW, avg_delay) =>
  writer.append(s"$dofW\t\t$avg_delay\n")
}
writer.append("\n")

// ------------------------------------------------------------------------------------------

// Number of Flights per Carrier
val carrierCounts = flightsRDD.map(flight => (flight.carrier, 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false)

carrierCounts.take(5).foreach { case (carrier, count) =>
  println(s"$carrier\t\t$count")
}

writer.append("Carrier\t\tNumber of Flights\n")
writer.append("-----------------------------\n")
carrierCounts.take(5).foreach { case (carrier, count) =>
  writer.append(s"$carrier\t\t$count\n")
}
writer.append("\n")

// Average Delay per Carrier
val carrierDelays = flightsRDD.map(flight => (flight.carrier, flight.depdelaymins)).groupByKey().mapValues(delays => delays.sum / delays.size.toDouble).sortBy(_._2, ascending = false)

carrierDelays.take(5).foreach { case (carrier, avgDelay) =>
  println(s"$carrier\t\t$avgDelay")
}

writer.append("Carrier\t\tAverage Delay (mins)\n")
writer.append("---------------------------------\n")
carrierDelays.take(5).foreach { case (carrier, avgDelay) =>
  writer.append(s"$carrier\t\t$avgDelay\n")
}
writer.append("\n")


// ------------------------------------------------------------------------------------------

// corelation between distance and speed total
val distance = flightsRDD.map(flight => flight.dist.toDouble)
val speed = flightsRDD.map(flight => flight.dist.toDouble / (flight.crselapsedtime.toDouble / 60))

val dist_to_speed_corr = Statistics.corr(distance, speed)

writer.append("\n\nCorrelation between distance and speed \n")
writer.append("------------------------------------------\n")
writer.append("The corelation between distance to speed is = " + dist_to_speed_corr.toString + "\n")

// corelation with respect to the arrival delay and departure delay
val arrivalDelay = flightsRDD.map(flight => flight.arrdelay.toDouble)
val departureDelay = flightsRDD.map(flight => flight.depdelaymins.toDouble)

val arrival_to_departure_corr = Statistics.corr(arrivalDelay, departureDelay)

writer.append("\n\nCorrelation between arrival delay and departure delay \n")
writer.append("------------------------------------------\n")
writer.append("The corelation between arrival delay and departure delay is = " + arrival_to_departure_corr.toString + "\n")
writer.append("\n")


// ------------------------------------------------------------------------------------------

// Average Speed per Carrier
val carrierSpeeds = flightsRDD.map(flight => (flight.carrier, flight.dist / (flight.crselapsedtime / 60.0))).groupByKey().mapValues(speeds => speeds.sum / speeds.size.toDouble).sortBy(_._2, ascending = false)

carrierSpeeds.take(5).foreach { case (carrier, averageSpeed) =>
  println(s"$carrier\t\t$averageSpeed")
}

writer.append("Carrier\t\tAverage Speed (miles per hour)\n")
writer.append("----------------------------------\n")
carrierSpeeds.take(5).foreach { case (carrier, averageSpeed) =>
  writer.append(s"$carrier\t\t$averageSpeed\n\n")
}

// Average Distance Covered per Carrier
val carrierDistances = flightsRDD.map(flight => (flight.carrier, flight.dist.toDouble)).groupByKey().mapValues(distances => distances.sum / distances.size).sortBy(_._2, ascending = false)

carrierDistances.take(5).foreach { case (carrier, averageDistance) =>
  println(s"$carrier\t\t$averageDistance")
}

writer.append("Carrier\t\tAverage Distance Covered (In miles)\n")
writer.append("-----------------------------------\n")
carrierDistances.take(5).foreach { case (carrier, averageDistance) =>
  writer.append(s"$carrier\t\t$averageDistance\n")
}
writer.append("\n")

// ------------------------------------------------------------------------------------------

// Hardworking planes
val flightsCountByTailNum = flightsRDD.map(flight => (flight.tailnum, 1)).reduceByKey(_ + _)

val averageSpeed = flightsRDD.map(flight => (flight.tailnum, flight.dist.toDouble / (flight.crselapsedtime.toDouble / 60))) // Find the average speed of each flight (tail number)

val avgSpeedByTailNum = averageSpeed.groupByKey().mapValues(speeds => speeds.sum / speeds.size) // Group by tail number and find the average speed

val stdDevByTailNum = averageSpeed.groupByKey().mapValues(speeds => {
  val avg = speeds.sum / speeds.size
  val squaredDiffs = speeds.map(speed => math.pow(speed - avg, 2))
  math.sqrt(squaredDiffs.sum / squaredDiffs.size)
})

val flightsInfo = flightsCountByTailNum.join(avgSpeedByTailNum).join(stdDevByTailNum)
val topFlights = flightsInfo.sortBy(_._2._1._1, ascending = false).take(5)


writer.append("\n\nTop planes based on flight count \n")
// Print the table header

writer.append("Tail Number\tCarrier\tAverage Speed\tStandard Deviation\tNumber of Flights\n")
writer.append("-------------------------------------------------------------------------\n")

// Print the top 5 flights
topFlights.foreach { case (tailnum, ((count, avgSpeed), stdDev)) =>
  val carrier = flightsRDD.filter(_.tailnum == tailnum).first().carrier
  writer.append(s"$tailnum\t\t$carrier\t$avgSpeed\t\t$stdDev\t\t$count\n")
}
writer.append("\n")


val tailNumberCounts = flightsRDD.map(flight => (flight.tailnum, 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false)
val topTailNumbers = tailNumberCounts.take(5)

// Calculate average and standard deviation for each plane
val tailNumberStats = topTailNumbers.map { case (tailnum, count) =>
  val tailNumberFlights = flightsRDD.filter(_.tailnum == tailnum)
  val average = tailNumberFlights.map(flight => flight.depdelaymins.toDouble).mean()
  val stdDev = tailNumberFlights.map(flight => flight.depdelaymins.toDouble).stdev()
  (tailnum, average, stdDev)
}

// Write the result to the file
writer.append("\nTop 5 Aircraft by Flight Count:\n")
tailNumberStats.foreach { case (tailnum, average, stdDev) =>
  writer.append(s"Aircraft: $tailnum\tAverage Departure Delay: $average minutes\tStandard Deviation: $stdDev minutes\n")
}

// ------------------------------------------------------------------------------------------

// Distance wise delay analysis

val very_short_haul = flightsRDD.filter(flight => flight.dist < 100)
val short_haul = flightsRDD.filter(flight => flight.dist >= 100 && flight.dist < 500)
val medium_haul = flightsRDD.filter(flight => flight.dist >= 500 && flight.dist < 1000)
val long_haul = flightsRDD.filter(flight => flight.dist >= 1000)

val avg_dep_delay_very_short_haul = very_short_haul.map(flight => flight.depdelaymins).sum() / very_short_haul.count()
val avg_dep_delay_short_haul = short_haul.map(flight => flight.depdelaymins).sum() / short_haul.count()
val avg_dep_delay_medium_haul = medium_haul.map(flight => flight.depdelaymins).sum() / medium_haul.count()
val avg_dep_delay_long_haul = long_haul.map(flight => flight.depdelaymins).sum() / long_haul.count()

writer.append("\n\nDeparture Delay by Distance Category\n")
writer.append("-------------------------------------\n")
writer.append(s"Very Short Haul: $avg_dep_delay_very_short_haul\n")
writer.append(s"Short Haul: $avg_dep_delay_short_haul\n")
writer.append(s"Medium Haul: $avg_dep_delay_medium_haul\n")
writer.append(s"Long Haul: $avg_dep_delay_long_haul\n")
writer.append("\n")

// ------------------------------------------------------------------------------------------

// time of day wise delay analysis

val morning_flights = flightsRDD.filter(flight => {
  val depHour = flight.deptime / 100  // Assuming deptime is in HHMM format
  depHour >= 6 && depHour <= 12
})

val afternoon_flights = flightsRDD.filter(flight => {
  val depHour = flight.deptime / 100
  depHour > 12 && depHour <= 18
})

val evening_flights = flightsRDD.filter(flight => {
  val depHour = flight.deptime / 100
  depHour > 18 && depHour <= 23
})

val late_night_flights = flightsRDD.filter(flight => {
  val depHour = flight.deptime / 100
  depHour > 23 || depHour <= 6
})

val morning_flights_count = morning_flights.count()
val afternoon_flights_count = afternoon_flights.count()
val evening_flights_count = evening_flights.count()
val late_night_flights_count = late_night_flights.count()

val delayed_morning_flights = morning_flights.filter(_.depdelaymins > 0)
val delayed_afternoon_flights = afternoon_flights.filter(_.depdelaymins > 0)
val delayed_evening_flights = evening_flights.filter(_.depdelaymins > 0)
val delayed_late_night_flights = late_night_flights.filter(_.depdelaymins > 0)

val delayed_morning_flights_count = delayed_morning_flights.count()
val delayed_afternoon_flights_count = delayed_afternoon_flights.count()
val delayed_evening_flights_count = delayed_evening_flights.count()
val delayed_late_night_flights_count = delayed_late_night_flights.count()

val avg_dep_delay_morning = delayed_morning_flights.map(_.depdelaymins).sum() / delayed_morning_flights_count.toDouble
val avg_dep_delay_afternoon = delayed_afternoon_flights.map(_.depdelaymins).sum() / delayed_afternoon_flights_count.toDouble
val avg_dep_delay_evening = delayed_evening_flights.map(_.depdelaymins).sum() / delayed_evening_flights_count.toDouble
val avg_dep_delay_late_night = delayed_late_night_flights.map(_.depdelaymins).sum() / delayed_late_night_flights_count.toDouble

val percent_morning_flights = (morning_flights_count.toDouble / totalFlights) * 100
val percent_afternoon_flights = (afternoon_flights_count.toDouble / totalFlights) * 100
val percent_evening_flights = (evening_flights_count.toDouble / totalFlights) * 100
val percent_late_night_flights = (late_night_flights_count.toDouble / totalFlights) * 100

val percent_delayed_morning_flights = (delayed_morning_flights_count.toDouble / morning_flights_count) * 100
val percent_delayed_afternoon_flights = (delayed_afternoon_flights_count.toDouble / afternoon_flights_count) * 100
val percent_delayed_evening_flights = (delayed_evening_flights_count.toDouble / evening_flights_count) * 100
val percent_delayed_late_night_flights = (delayed_late_night_flights_count.toDouble / late_night_flights_count) * 100

writer.append("Time of Day\tTotal Flights (%)\tDelayed Flights (%)\tAverage Delay (minutes)\n")
writer.append("----------------------------------------------------------------------------\n")
writer.append(f"Morning\t$percent_morning_flights%.2f%%\t$percent_delayed_morning_flights%.2f%%\t$avg_dep_delay_morning%.2f\n")
writer.append(f"Afternoon\t$percent_afternoon_flights%.2f%%\t$percent_delayed_afternoon_flights%.2f%%\t$avg_dep_delay_afternoon%.2f\n")
writer.append(f"Evening\t$percent_evening_flights%.2f%%\t$percent_delayed_evening_flights%.2f%%\t$avg_dep_delay_evening%.2f\n")
writer.append(f"Late Night\t$percent_late_night_flights%.2f%%\t$percent_delayed_late_night_flights%.2f%%\t$avg_dep_delay_late_night%.2f\n")
writer.append("\n")

// ------------------------------------------------------------------------------------------


// Number of planes arriving at each airport - Top 5
val arrivingCounts = flightsRDD.map(flight => (flight.dest, 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false)


println("Number of planes arriving at each airport - Top 5")
arrivingCounts.take(5).foreach { case (dest, count) =>
  println(s"$dest\t\t$count")
}

writer.append("Number of planes arriving at each airport - Top 5\n")
writer.append("Destination\tCount\n")
writer.append("---------------------\n")
arrivingCounts.take(5).foreach { case (dest, count) =>
  writer.append(s"$dest\t\t$count\n\n")
}

// Num of planes departing from each airport - Top 5
val departingCounts = flightsRDD.map(flight => (flight.origin, 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false)


println("Number of planes departing from each airport - Top 5")
departingCounts.take(5).foreach { case (origin, count) =>
  println(s"$origin\t\t$count")
}

writer.append("Number of planes departing from each airport - Top 5\n")
writer.append("Origin\t\tCount\n")
writer.append("---------------------\n")
departingCounts.take(5).foreach { case (origin, count) =>
  writer.append(s"$origin\t\t$count\n\n")
}


// Delay in the busiest cities
val busiestCities = flightsRDD.map(flight => (flight.dest, flight.arrdelay)).reduceByKey(_ + _).sortBy(_._2, ascending = false)
val topBusyCities = busiestCities.take(5)

println("Top 5 Busiest Cities by Delay:")
topBusyCities.foreach { case (city, delay) =>
  println(s"$city\t\t$delay Minutes")
}

writer.append("Top 5 Busiest Cities by Delay:\n")
writer.append("---------------------\n")
topBusyCities.foreach { case (city, delay) =>
  writer.append(s"$city\t\t$delay Minutes\n\n")
}

// ------------------------------------------------------------------------------------------

// count by routes
// ------------------------------------------------------------------------------------------

val routeCounts = flightsRDD.map(flight => (flight.origin + "-" + flight.dest, 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false)

val topRoutes = routeCounts.take(10) // Change the number to get more or fewer routes

writer.append("\n\nTop Flight Routes\n")
writer.append("-----------------\n")
topRoutes.foreach { case (route, count) =>
  val routeFlights = flightsRDD.filter(flight => flight.origin + "-" + flight.dest == route)
  val routeDelayedFlights = routeFlights.filter(_.depdelaymins > 0)
  val avgDepDelay = routeDelayedFlights.map(_.depdelaymins).sum() / routeDelayedFlights.count().toDouble

  val routeDistances = routeFlights.map(_.dist.toDouble)
  val avgDistance = routeDistances.sum() / routeDistances.count()

  writer.append(s"$route: $count flights\tAverage Delay: $avgDepDelay minutes\tDistance: $avgDistance\n")
}

// ------------------------------------------------------------------------------------------

val speedData  = flightsRDD.map(flight => flight.dist.toDouble / (flight.crselapsedtime/60))

val meanSpeed = speedData.mean()
val varianceSpeed = speedData.map(speed => math.pow(speed - meanSpeed, 2)).mean()
val stdDevSpeed = math.sqrt(varianceSpeed)
writer.append(s"Standard Deviation of Speed: $stdDevSpeed\n")

// ------------------------------------------------------------------------------------------


writer.close()