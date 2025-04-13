#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sr.h"

#define WINDOW_SIZE 8
#define TIMEOUT 20.0
#define MAX_SEQ (2 * WINDOW_SIZE) // Sequence number space

/* Utility Functions */
struct pkt make_packet(int seqnum, int acknum, char *payload);
void send_packet(int AorB, struct pkt packet);
int is_in_window(int base, int seqnum);

/* ---------- Sender A ---------- */

static int base_A = 0;
static int nextseqnum_A = 0;
static struct pkt window_A[WINDOW_SIZE];
static int acked[WINDOW_SIZE];
static float timers[WINDOW_SIZE]; // Simplified timer tracking

void A_output(struct msg message) {
    if ((nextseqnum_A + MAX_SEQ - base_A) % MAX_SEQ < WINDOW_SIZE) {
        struct pkt packet = make_packet(nextseqnum_A, -1, message.data);
        window_A[nextseqnum_A % WINDOW_SIZE] = packet;
        acked[nextseqnum_A % WINDOW_SIZE] = 0;
        send_packet(A, packet);
        starttimer(A, TIMEOUT);
        timers[nextseqnum_A % WINDOW_SIZE] = TIMEOUT;
        printf("[A] Sent packet %d\n", nextseqnum_A);
        nextseqnum_A = (nextseqnum_A + 1) % MAX_SEQ;
    } else {
        printf("[A] Window full. Message dropped.\n");
    }
}

void A_input(struct pkt packet) {
    if (packet.acknum < 0 || packet.acknum >= MAX_SEQ) return;
    int index = packet.acknum % WINDOW_SIZE;
    acked[index] = 1;
    printf("[A] Received ACK %d\n", packet.acknum);

    // Slide window forward
    while (acked[base_A % WINDOW_SIZE]) {
        acked[base_A % WINDOW_SIZE] = 0;
        timers[base_A % WINDOW_SIZE] = 0;
        base_A = (base_A + 1) % MAX_SEQ;
    }
}

void A_timerinterrupt() {
    // Retransmit all unACKed and timed-out packets
    for (int i = 0; i < WINDOW_SIZE; i++) {
        if (!acked[i] && timers[i] > 0) {
            send_packet(A, window_A[i]);
            starttimer(A, TIMEOUT);
            timers[i] = TIMEOUT;
            printf("[A] Timeout: Retransmitting packet %d\n", window_A[i].seqnum);
        }
    }
}

void A_init() {
    base_A = 0;
    nextseqnum_A = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        acked[i] = 0;
        timers[i] = 0;
    }
    printf("[A] Initialized.\n");
}

/* ---------- Receiver B ---------- */

static int base_B = 0;
static struct pkt buffer_B[WINDOW_SIZE];
static int received[WINDOW_SIZE];

void B_input(struct pkt packet) {
    if (packet.seqnum < 0 || packet.seqnum >= MAX_SEQ) return;

    int index = packet.seqnum % WINDOW_SIZE;

    if (is_in_window(base_B, packet.seqnum)) {
        if (!received[index]) {
            buffer_B[index] = packet;
            received[index] = 1;
            tolayer5(B, packet.payload);
            printf("[B] Received and buffered packet %d\n", packet.seqnum);
        } else {
            printf("[B] Duplicate packet %d received\n", packet.seqnum);
        }

        // Send ACK
        struct pkt ackpkt = make_packet(0, packet.seqnum, NULL);
        send_packet(B, ackpkt);
        printf("[B] Sent ACK %d\n", packet.seqnum);

        // Deliver in-order packets
        while (received[base_B % WINDOW_SIZE]) {
            received[base_B % WINDOW_SIZE] = 0;
            base_B = (base_B + 1) % MAX_SEQ;
        }
    } else {
        // Out-of-window packet; still ACK
        struct pkt ackpkt = make_packet(0, packet.seqnum, NULL);
        send_packet(B, ackpkt);
        printf("[B] Out-of-window packet %d, ACK sent\n", packet.seqnum);
    }
}

void B_init() {
    base_B = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        received[i] = 0;
    }
    printf("[B] Initialized.\n");
}

/* ---------- Utility Functions ---------- */

struct pkt make_packet(int seqnum, int acknum, char *payload) {
    struct pkt packet;
    packet.seqnum = seqnum;
    packet.acknum = acknum;
    memset(packet.payload, 0, sizeof(packet.payload));
    if (payload != NULL)
        strncpy(packet.payload, payload, 20);
    packet.checksum = seqnum + acknum;
    for (int i = 0; i < 20; i++) {
        packet.checksum += packet.payload[i];
    }
    return packet;
}

void send_packet(int AorB, struct pkt packet) {
    tolayer3(AorB, packet);
}

int is_in_window(int base, int seqnum) {
    if (((seqnum + MAX_SEQ - base) % MAX_SEQ) < WINDOW_SIZE)
        return 1;
    return 0;
}
