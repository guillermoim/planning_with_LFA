; Automatically converted to only require STRIPS and negative preconditions

(define (problem transport-city-sequential-27nodes-1000size-4degree-100mindistance-3trucks-10packages-2008seed)
  (:domain transport)
  (:objects city-loc-1 city-loc-2 city-loc-3 city-loc-4 city-loc-5 city-loc-6 city-loc-7 city-loc-8 city-loc-9 city-loc-10 city-loc-11 city-loc-12 city-loc-13 city-loc-14 city-loc-15 city-loc-16 city-loc-17 city-loc-18 city-loc-19 city-loc-20 city-loc-21 city-loc-22 city-loc-23 city-loc-24 city-loc-25 city-loc-26 city-loc-27 truck-1 truck-2 truck-3 package-1 package-2 package-3 package-4 package-5 package-6 package-7 package-8 package-9 package-10 capacity-0 capacity-1 capacity-2 capacity-3 capacity-4)
  (:init
    (capacity-predecessor capacity-0 capacity-1)
    (capacity-predecessor capacity-1 capacity-2)
    (capacity-predecessor capacity-2 capacity-3)
    (capacity-predecessor capacity-3 capacity-4)
    (road city-loc-4 city-loc-2)
    (road city-loc-2 city-loc-4)
    (road city-loc-7 city-loc-5)
    (road city-loc-5 city-loc-7)
    (road city-loc-8 city-loc-2)
    (road city-loc-2 city-loc-8)
    (road city-loc-8 city-loc-5)
    (road city-loc-5 city-loc-8)
    (road city-loc-8 city-loc-7)
    (road city-loc-7 city-loc-8)
    (road city-loc-9 city-loc-2)
    (road city-loc-2 city-loc-9)
    (road city-loc-9 city-loc-4)
    (road city-loc-4 city-loc-9)
    (road city-loc-9 city-loc-5)
    (road city-loc-5 city-loc-9)
    (road city-loc-9 city-loc-8)
    (road city-loc-8 city-loc-9)
    (road city-loc-10 city-loc-4)
    (road city-loc-4 city-loc-10)
    (road city-loc-11 city-loc-9)
    (road city-loc-9 city-loc-11)
    (road city-loc-12 city-loc-4)
    (road city-loc-4 city-loc-12)
    (road city-loc-12 city-loc-10)
    (road city-loc-10 city-loc-12)
    (road city-loc-13 city-loc-11)
    (road city-loc-11 city-loc-13)
    (road city-loc-13 city-loc-12)
    (road city-loc-12 city-loc-13)
    (road city-loc-14 city-loc-2)
    (road city-loc-2 city-loc-14)
    (road city-loc-14 city-loc-3)
    (road city-loc-3 city-loc-14)
    (road city-loc-14 city-loc-4)
    (road city-loc-4 city-loc-14)
    (road city-loc-14 city-loc-10)
    (road city-loc-10 city-loc-14)
    (road city-loc-14 city-loc-12)
    (road city-loc-12 city-loc-14)
    (road city-loc-15 city-loc-3)
    (road city-loc-3 city-loc-15)
    (road city-loc-16 city-loc-5)
    (road city-loc-5 city-loc-16)
    (road city-loc-16 city-loc-9)
    (road city-loc-9 city-loc-16)
    (road city-loc-16 city-loc-11)
    (road city-loc-11 city-loc-16)
    (road city-loc-18 city-loc-7)
    (road city-loc-7 city-loc-18)
    (road city-loc-18 city-loc-8)
    (road city-loc-8 city-loc-18)
    (road city-loc-18 city-loc-17)
    (road city-loc-17 city-loc-18)
    (road city-loc-19 city-loc-6)
    (road city-loc-6 city-loc-19)
    (road city-loc-20 city-loc-17)
    (road city-loc-17 city-loc-20)
    (road city-loc-20 city-loc-18)
    (road city-loc-18 city-loc-20)
    (road city-loc-21 city-loc-3)
    (road city-loc-3 city-loc-21)
    (road city-loc-21 city-loc-10)
    (road city-loc-10 city-loc-21)
    (road city-loc-21 city-loc-12)
    (road city-loc-12 city-loc-21)
    (road city-loc-21 city-loc-14)
    (road city-loc-14 city-loc-21)
    (road city-loc-22 city-loc-3)
    (road city-loc-3 city-loc-22)
    (road city-loc-22 city-loc-6)
    (road city-loc-6 city-loc-22)
    (road city-loc-22 city-loc-15)
    (road city-loc-15 city-loc-22)
    (road city-loc-23 city-loc-2)
    (road city-loc-2 city-loc-23)
    (road city-loc-23 city-loc-3)
    (road city-loc-3 city-loc-23)
    (road city-loc-23 city-loc-4)
    (road city-loc-4 city-loc-23)
    (road city-loc-23 city-loc-14)
    (road city-loc-14 city-loc-23)
    (road city-loc-23 city-loc-15)
    (road city-loc-15 city-loc-23)
    (road city-loc-24 city-loc-6)
    (road city-loc-6 city-loc-24)
    (road city-loc-24 city-loc-19)
    (road city-loc-19 city-loc-24)
    (road city-loc-25 city-loc-11)
    (road city-loc-11 city-loc-25)
    (road city-loc-25 city-loc-16)
    (road city-loc-16 city-loc-25)
    (road city-loc-26 city-loc-3)
    (road city-loc-3 city-loc-26)
    (road city-loc-26 city-loc-6)
    (road city-loc-6 city-loc-26)
    (road city-loc-26 city-loc-15)
    (road city-loc-15 city-loc-26)
    (road city-loc-26 city-loc-19)
    (road city-loc-19 city-loc-26)
    (road city-loc-26 city-loc-22)
    (road city-loc-22 city-loc-26)
    (road city-loc-26 city-loc-24)
    (road city-loc-24 city-loc-26)
    (road city-loc-27 city-loc-1)
    (road city-loc-1 city-loc-27)
    (road city-loc-27 city-loc-15)
    (road city-loc-15 city-loc-27)
    (road city-loc-27 city-loc-23)
    (road city-loc-23 city-loc-27)
    (at package-1 city-loc-10)
    (at package-2 city-loc-1)
    (at package-3 city-loc-10)
    (at package-4 city-loc-2)
    (at package-5 city-loc-6)
    (at package-6 city-loc-7)
    (at package-7 city-loc-25)
    (at package-8 city-loc-24)
    (at package-9 city-loc-18)
    (at package-10 city-loc-22)
    (at truck-1 city-loc-15)
    (capacity truck-1 capacity-3)
    (at truck-2 city-loc-27)
    (capacity truck-2 capacity-3)
    (at truck-3 city-loc-18)
    (capacity truck-3 capacity-2)
    (location city-loc-1)
    (location city-loc-2)
    (location city-loc-3)
    (location city-loc-4)
    (location city-loc-5)
    (location city-loc-6)
    (location city-loc-7)
    (location city-loc-8)
    (location city-loc-9)
    (location city-loc-10)
    (location city-loc-11)
    (location city-loc-12)
    (location city-loc-13)
    (location city-loc-14)
    (location city-loc-15)
    (location city-loc-16)
    (location city-loc-17)
    (location city-loc-18)
    (location city-loc-19)
    (location city-loc-20)
    (location city-loc-21)
    (location city-loc-22)
    (location city-loc-23)
    (location city-loc-24)
    (location city-loc-25)
    (location city-loc-26)
    (location city-loc-27)
    (vehicle truck-1)
    (locatable truck-1)
    (vehicle truck-2)
    (locatable truck-2)
    (vehicle truck-3)
    (locatable truck-3)
    (package package-1)
    (locatable package-1)
    (package package-2)
    (locatable package-2)
    (package package-3)
    (locatable package-3)
    (package package-4)
    (locatable package-4)
    (package package-5)
    (locatable package-5)
    (package package-6)
    (locatable package-6)
    (package package-7)
    (locatable package-7)
    (package package-8)
    (locatable package-8)
    (package package-9)
    (locatable package-9)
    (package package-10)
    (locatable package-10)
    (capacity-number capacity-0)
    (capacity-number capacity-1)
    (capacity-number capacity-2)
    (capacity-number capacity-3)
    (capacity-number capacity-4)
  )
  (:goal
    (and
      (at package-1 city-loc-15)
    )
  )
)
