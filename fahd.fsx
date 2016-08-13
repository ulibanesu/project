#r "FSharp.Data.2.3.0/bin/FSharp.Data.dll"

open System
open System.IO
open System.Threading
open FSharp.Data
open System.Net
open System.IO

type Test = JsonProvider<"/Users/KHATIBUSINESS/bitcoin/antoine0.json">


let parcours (t1:Test.Asks) (t2:Test.Asks) (ok:int) =
  let mutable ctr = ok
  [for i in 0..(t1.L.Length-1) do
    if (ctr>0) then
      let mutable b = 0.0M
      let mutable c = 0.0M
      let mutable counter = true
      let mutable d = 0
      for ev in t1.L.[i].L do
        match ev.S  with
          |Some s ->  if counter then
                        b <- s
                        counter <- not(counter)
                      else
                        c <- s
          |None -> ()
        match ev.N with
          |Some n -> d <- n
          |None -> ()
      yield (b,c,d)
      ctr <- ctr - 1

      let mutable e = 0.0M
      let mutable f = 0.0M
      let mutable counter = true
      let mutable g = 0
      for ev in t2.L.[i].L do
        match ev.S  with
          |Some s ->  if counter then
                        e <- s
                        counter <- not(counter)
                      else
                        f <- s
          |None -> ()
        match ev.N with
          |Some n -> g <- n
          |None -> ()
      yield (e,f,g)
  ]


let realparcours (tableau:JsonProvider<"/Users/KHATIBUSINESS/bitcoin/antoine0.json">.Root []) (n:int) =
  [for elem in tableau do
      yield parcours elem.Result.M.Xxbtzeur.M.Bids elem.Result.M.Xxbtzeur.M.Asks n
  ]

let printer truc =
  let (a,b,c) = truc
  a.ToString() + " | " + b.ToString() + " | " + c.ToString()

let afficher aList =
  let rec affichage liste s =
    match liste with
    |[] -> s
    |[hd] -> s + (printer hd)
    |hd1::hd2::tl -> affichage tl (s + (printer hd1) + " || " + (printer hd2) + Environment.NewLine)
  in affichage aList ""

let beautify (aList:('a * 'b * 'c) list List) =
 let size = aList.Length
 let mutable s = Array.create size ""
 let rec beau (liste:('a * 'b * 'c) list List) counter =
    match liste with
      |[] -> s
      |hd::tl -> s.[counter] <- (afficher  hd)
                 beau tl (counter + 1)
  in beau aList 0

let olivier n premiers =
      let arr1 = (Array.create (n+1) [|""|])
      let arr2 = (Array.create (n+1) [|""|])
      let liste = [for i=0 to n do
                      let s = "/Users/KHATIBUSINESS/bitcoin/antoine"+i.ToString()+".json"
                      yield Test.Load(s)]
      let rec aux (aListe:JsonProvider<"/Users/KHATIBUSINESS/bitcoin/antoine0.json">.Root [] list) (s1:string [] []) (s2:string [] []) counter =
        match aListe with
          |[] -> ()
          |hd::tl -> s1.[counter] <- hd.Length |> realparcours hd |> beautify
                     s2.[counter] <- premiers |> realparcours hd |> beautify
                     aux tl s1 s2 (counter + 1)
      in aux liste arr1 arr2 0
      for elem in arr1 do
          File.AppendAllLines("/Users/KHATIBUSINESS/bitcoin/test1.txt",elem)
      for elem in arr2 do
          File.AppendAllLines("/Users/KHATIBUSINESS/bitcoin/test2.txt",elem)

olivier 7 15
