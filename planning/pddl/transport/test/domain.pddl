; Automatically converted to only require STRIPS and negative preconditions

(define (domain transport)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (location ?x)
    (target ?x)
    (locatable ?x)
    (vehicle ?x)
    (package ?x)
    (capacity-number ?x)
    (road ?l1 ?l2)
    (at ?x ?v)
    (in ?x ?v)
    (capacity ?v ?s1)
    (capacity-predecessor ?s1 ?s2)
  )

  (:action drive
    :parameters (?v ?l1 ?l2)
    :precondition (and
      (vehicle ?v)
      (location ?l1)
      (location ?l2)
      (at ?v ?l1)
      (road ?l1 ?l2)
    )
    :effect (and
      (not (at ?v ?l1))
      (at ?v ?l2)
    )
  )

  (:action pick-up
    :parameters (?v ?l ?p ?s1 ?s2)
    :precondition (and
      (vehicle ?v)
      (location ?l)
      (package ?p)
      (capacity-number ?s1)
      (capacity-number ?s2)
      (at ?v ?l)
      (at ?p ?l)
      (capacity-predecessor ?s1 ?s2)
      (capacity ?v ?s2)
    )
    :effect (and
      (not (at ?p ?l))
      (in ?p ?v)
      (capacity ?v ?s1)
      (not (capacity ?v ?s2))
    )
  )

  (:action drop
    :parameters (?v ?l ?p ?s1 ?s2)
    :precondition (and
      (vehicle ?v)
      (location ?l)
      (package ?p)
      (capacity-number ?s1)
      (capacity-number ?s2)
      (at ?v ?l)
      (in ?p ?v)
      (capacity-predecessor ?s1 ?s2)
      (capacity ?v ?s1)
    )
    :effect (and
      (not (in ?p ?v))
      (at ?p ?l)
      (capacity ?v ?s2)
      (not (capacity ?v ?s1))
    )
  )
)
